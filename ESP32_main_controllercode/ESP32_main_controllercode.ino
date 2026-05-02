/*
 * =============================================================================
 *  RD-03D Multi-Target Radar Parser for ESP32 DevKit V1
 *  -------------------------------------------------------------
 *  Bachelor's thesis project: RF motion sensor based on ESP32 + RD-03D
 *  Author:    XYU
 *  Date:      2026
 *
 *  Hardware:
 *      - ESP32 DevKit V1 (module ESP-WROOM-32)
 *      - Ai-Thinker RD-03D 24-GHz FMCW radar
 *
 *  Wiring (cross-over UART2):
 *      RD-03D 5V  -- 5V (VIN)        ESP32 DevKit V1
 *      RD-03D GND -- GND
 *      RD-03D TX  -- GPIO16 (RX2)
 *      RD-03D RX  -- GPIO17 (TX2)
 *
 *  Architecture:
 *      Core 1 -> uartTask (UART receiver + frame parser)
 *      Core 0 -> publisherTask (consumer: Serial print, MQTT, etc.)
 *      Communication via FreeRTOS queue (overwrite mode = always latest data).
 *
 *  Frame format (30 bytes, RD-03D multi-target mode):
 *      [0..3]   header   = AA FF 03 00
 *      [4..11]  target 1 = X(2) Y(2) V(2) D(2)
 *      [12..19] target 2
 *      [20..27] target 3
 *      [28..29] footer   = 55 CC
 *
 *      X, Y, V are signed 16-bit "sign-magnitude" (bit15 = sign, bits14..0 = abs)
 *      D       is unsigned 16-bit (resolution range)
 *      All multi-byte fields are little-endian.
 *      An empty target slot is filled with zero bytes.
 * =============================================================================
 */

#include <Arduino.h>

// ========================== Compile-time configuration ======================
namespace cfg {
    constexpr uint8_t  RX_PIN          = 16;            // ESP32 GPIO16 <- RD-03D TX
    constexpr uint8_t  TX_PIN          = 17;            // ESP32 GPIO17 -> RD-03D RX
    constexpr uint32_t BAUD_RATE       = 256000;        // RD-03D fixed baud rate
    constexpr uint16_t UART_RX_BUF_LEN = 512;           // ESP32 UART driver buffer
    constexpr uint16_t FRAME_LEN       = 30;            // total bytes in one frame
    constexpr uint8_t  MAX_TARGETS     = 3;             // RD-03D tracks up to 3
    constexpr uint32_t LOG_PERIOD_MS   = 100;           // console print throttle
    constexpr BaseType_t UART_CORE     = 1;             // pin uartTask to core 1
    constexpr BaseType_t PUB_CORE      = 0;             // pin publisher to core 0
}

// ============================ Public data types =============================
struct Target {
    int16_t  x_mm;          // X coordinate, mm  (-: left of normal, +: right)
    int16_t  y_mm;          // Y coordinate, mm  (range from radar, signed)
    int16_t  speed_cm_s;    // radial speed, cm/s (-: away, +: towards)
    uint16_t distance_mm;   // resolution distance, mm
    bool     present;       // true if slot is non-empty
};

struct RadarFrame {
    Target   targets[cfg::MAX_TARGETS];
    uint32_t timestamp_ms;  // millis() at the moment of parsing
    uint32_t frame_number;  // monotonic counter, useful for diagnostics
};

// ============================ Module-private data ===========================
namespace {

QueueHandle_t s_radarQueue = nullptr;   // length=1, overwrite mode -> always latest

// Single-target / Multi-target detection commands (per Ai-Thinker AT manual)
constexpr uint8_t MULTI_TARGET_CMD[12] = {
    0xFD, 0xFC, 0xFB, 0xFA,
    0x02, 0x00, 0x90, 0x00,
    0x04, 0x03, 0x02, 0x01
};

// ---------------------------------------------------------------------------
// Decode a 16-bit sign-magnitude value as transmitted by RD-03D.
// Per RD-03D reference parser: bit15 = 1 -> positive,  bit15 = 0 -> negative.
// (Verified empirically; community parsers use the same convention.)
// ---------------------------------------------------------------------------
static inline int16_t decodeSignMagnitude(uint8_t lo, uint8_t hi) {
    const uint16_t raw = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
    const int16_t  mag = static_cast<int16_t>(raw & 0x7FFF);
    return (raw & 0x8000) ? mag : -mag;
}

static inline uint16_t decodeUnsigned(uint8_t lo, uint8_t hi) {
    return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
}

// ---------------------------------------------------------------------------
// Parse one validated 30-byte frame into a RadarFrame structure.
// ---------------------------------------------------------------------------
static void parseFrame(const uint8_t* buf, RadarFrame& out) {
    for (uint8_t i = 0; i < cfg::MAX_TARGETS; ++i) {
        const uint8_t* p = buf + 4 + i * 8;
        Target& t = out.targets[i];

        t.x_mm        = decodeSignMagnitude(p[0], p[1]);
        t.y_mm        = decodeSignMagnitude(p[2], p[3]);
        t.speed_cm_s  = decodeSignMagnitude(p[4], p[5]);
        t.distance_mm = decodeUnsigned    (p[6], p[7]);

        // Empty slot: all 8 bytes are zero -> mark as not present
        t.present = !(p[0]==0 && p[1]==0 && p[2]==0 && p[3]==0 &&
                      p[4]==0 && p[5]==0 && p[6]==0 && p[7]==0);
    }
}

// ---------------------------------------------------------------------------
// UART receive task — runs on dedicated core, never blocks consumers.
// Implements a small finite-state machine that synchronises by:
//   1) the 4-byte header signature  AA FF 03 00
//   2) checks frame length (30 bytes)
//   3) verifies the 2-byte footer   55 CC
// On any mismatch -> drops the buffer and re-syncs (no silent corruption).
// ---------------------------------------------------------------------------
static void uartTask(void* /*pv*/) {
    enum class State : uint8_t { WaitHdr0, WaitHdr1, WaitHdr2, WaitHdr3, ReadBody };

    static uint8_t  body[cfg::FRAME_LEN];
    State    state    = State::WaitHdr0;
    uint16_t bodyIdx  = 0;
    uint32_t frameNum = 0;

    for (;;) {
        // Read up to whatever is available; non-blocking inside the inner loop.
        while (Serial1.available() > 0) {
            const uint8_t b = static_cast<uint8_t>(Serial1.read());

            switch (state) {
                case State::WaitHdr0:
                    if (b == 0xAA) { body[0] = b; state = State::WaitHdr1; }
                    break;

                case State::WaitHdr1:
                    if (b == 0xFF) { body[1] = b; state = State::WaitHdr2; }
                    else           { state = (b == 0xAA) ? State::WaitHdr1 : State::WaitHdr0; }
                    break;

                case State::WaitHdr2:
                    if (b == 0x03) { body[2] = b; state = State::WaitHdr3; }
                    else           { state = (b == 0xAA) ? State::WaitHdr1 : State::WaitHdr0; }
                    break;

                case State::WaitHdr3:
                    if (b == 0x00) { body[3] = b; bodyIdx = 4; state = State::ReadBody; }
                    else           { state = (b == 0xAA) ? State::WaitHdr1 : State::WaitHdr0; }
                    break;

                case State::ReadBody:
                    body[bodyIdx++] = b;
                    if (bodyIdx >= cfg::FRAME_LEN) {
                        // Validate footer
                        if (body[28] == 0x55 && body[29] == 0xCC) {
                            RadarFrame f;
                            parseFrame(body, f);
                            f.timestamp_ms = millis();
                            f.frame_number = ++frameNum;
                            // Overwrite-mode queue: consumer always sees latest frame
                            xQueueOverwrite(s_radarQueue, &f);
                        }
                        // On both success and failure: re-sync
                        bodyIdx = 0;
                        state   = State::WaitHdr0;
                    }
                    break;
            }
        }
        // Yield ~1 tick to let other tasks run; UART driver buffers bytes during sleep.
        vTaskDelay(pdMS_TO_TICKS(2));
    }
}

// ---------------------------------------------------------------------------
// Consumer task — reads the latest frame and prints throttled human-readable
// log to USB serial. Replace body with MQTT publish, BLE notify, etc.
// ---------------------------------------------------------------------------
static void publisherTask(void* /*pv*/) {
    RadarFrame f;
    uint32_t lastPrintMs = 0;

    for (;;) {
        if (xQueuePeek(s_radarQueue, &f, pdMS_TO_TICKS(50)) == pdTRUE) {
            const uint32_t now = millis();
            if (now - lastPrintMs >= cfg::LOG_PERIOD_MS) {
                lastPrintMs = now;

                // Compact JSON, one frame per line
                Serial.printf("{\"n\":%lu,\"t\":%lu,\"targets\":[",
                              f.frame_number, f.timestamp_ms);

                bool first = true;
                for (uint8_t i = 0; i < cfg::MAX_TARGETS; ++i) {
                    const Target& t = f.targets[i];
                    if (!t.present) continue;

                    if (!first) Serial.print(',');
                    first = false;

                    const float dist_m  = t.distance_mm / 1000.0f;
                    const float angle   = atan2f(static_cast<float>(t.x_mm),
                                                  static_cast<float>(t.y_mm)) * 57.2958f;
                    const float speed_m = t.speed_cm_s / 100.0f;

                    Serial.printf("{\"id\":%u,\"r\":%.3f,\"a\":%.2f,\"v\":%.3f,"
                                  "\"x\":%d,\"y\":%d}",
                                  i + 1, dist_m, angle, speed_m, t.x_mm, t.y_mm);
                }
                Serial.println("]}");
            }
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

} // anonymous namespace

// ================================ Setup =====================================
void setup() {
    Serial.begin(115200);
    delay(50);
    Serial.println();
    Serial.println(F("RD-03D Multi-Target Radar — ESP32 firmware v1.0"));

    // Enlarge UART driver RX buffer to tolerate scheduler latency at 256 kbps
    Serial1.setRxBufferSize(cfg::UART_RX_BUF_LEN);
    Serial1.begin(cfg::BAUD_RATE, SERIAL_8N1, cfg::RX_PIN, cfg::TX_PIN);

    // Send command to switch radar into multi-target tracking mode
    delay(200);
    Serial1.write(MULTI_TARGET_CMD, sizeof(MULTI_TARGET_CMD));
    delay(50);
    Serial.println(F("Multi-target detection mode requested."));

    // Length-1 overwrite queue: consumer always sees the freshest frame,
    // older frames are discarded — appropriate for periodic telemetry.
    s_radarQueue = xQueueCreate(1, sizeof(RadarFrame));
    if (s_radarQueue == nullptr) {
        Serial.println(F("FATAL: cannot allocate radar queue"));
        while (true) { delay(1000); }
    }

    BaseType_t ok;
    ok = xTaskCreatePinnedToCore(uartTask,      "uartTask",      4096, nullptr, 10, nullptr, cfg::UART_CORE);
    configASSERT(ok == pdPASS);

    ok = xTaskCreatePinnedToCore(publisherTask, "publisherTask", 4096, nullptr,  1, nullptr, cfg::PUB_CORE);
    configASSERT(ok == pdPASS);

    Serial.println(F("Tasks started.  Reading radar..."));
}

// ================================ Loop ======================================
void loop() {
    // Intentionally empty.
    // Application logic (Wi-Fi connection, CSI processing, MQTT publish,
    // OTA updates, etc.) is implemented as additional FreeRTOS tasks
    // that consume RadarFrame via xQueuePeek(s_radarQueue, ...).
    vTaskDelay(pdMS_TO_TICKS(1000));
}
