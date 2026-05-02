"""
=============================================================================
  RD-03D multi-target radar visualizer
  -------------------------------------------------------------
  Bachelor's thesis project: RF motion sensor based on ESP32 + RD-03D

  Reads JSON frames from ESP32 over serial:
      {"n":42,"t":3815,"targets":[
          {"id":1,"r":0.36,"a":-18.0,"v":0.0,"x":-111,"y":340},
          {"id":2,"r":1.20,"a": 30.0,"v":-0.2,"x":600, "y":1040}
      ]}

  Renders a polar radar-style plot with:
    - range rings labelled in metres
    - up to 3 targets displayed simultaneously, colour-coded by id
    - short trail of recent positions (fading) for each target
    - on-screen telemetry panel
    - auto-clears targets that disappear

  Requirements:
      pip install pyserial matplotlib numpy

  Usage:
      python radar_viz.py            # auto-detects port
      python radar_viz.py COM7       # explicit port
      python radar_viz.py --port /dev/ttyUSB0 --baud 115200
=============================================================================
"""

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import serial
import serial.tools.list_ports


# ============================== Configuration ===============================

DEFAULT_BAUD     = 115200
MAX_RANGE_M      = 5.0          # display range (radar spec is 8 m, but 5 m is realistic indoors)
MAX_TARGETS      = 3
TRAIL_LEN        = 20           # number of recent positions kept per target
TARGET_TIMEOUT_S = 0.5          # if no update in this time -> target disappears
COLORS           = ['#ff3030', '#30c0ff', '#ffd030']   # red / cyan / yellow
RING_STEP_M      = 1.0          # range ring step
ANIM_INTERVAL_MS = 50           # plot refresh period


# ============================ Data structures ===============================

@dataclass
class TargetState:
    """Live state of a single radar target."""
    id: int
    r_m: float = 0.0
    angle_deg: float = 0.0
    speed_m_s: float = 0.0
    last_seen: float = 0.0
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))


# ============================ Serial reader thread ==========================

class SerialReader(threading.Thread):
    """
    Background thread: reads JSON frames from ESP32, updates shared state.
    Decoupled from the GUI to avoid blocking the matplotlib main loop.
    """
    def __init__(self, port: str, baud: int, state_lock: threading.Lock,
                 targets: dict, stats: dict):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.lock = state_lock
        self.targets = targets   # id -> TargetState
        self.stats = stats       # frame counters etc.
        self.stop_flag = False

    def run(self) -> None:
        try:
            ser = serial.Serial(self.port, self.baud, timeout=1.0)
        except serial.SerialException as e:
            print(f"[ERROR] Cannot open {self.port}: {e}", file=sys.stderr)
            return

        print(f"[INFO] Connected to {self.port} @ {self.baud} bps")
        ser.reset_input_buffer()

        while not self.stop_flag:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
            except serial.SerialException as e:
                print(f"[ERROR] Serial read failed: {e}", file=sys.stderr)
                break

            if not line or not line.startswith('{'):
                continue

            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                # Malformed line (boot banner, partial buffer, ...) -> skip silently
                continue

            self._update(frame)

        ser.close()

    def _update(self, frame: dict) -> None:
        now = time.time()
        with self.lock:
            self.stats['frames'] += 1
            self.stats['last_n'] = frame.get('n', 0)
            self.stats['last_ms'] = frame.get('t', 0)

            for tgt in frame.get('targets', []):
                tid = tgt['id']
                st = self.targets.setdefault(tid, TargetState(id=tid))

                # Use precise (x, y) from radar instead of quantised distance
                x_m = tgt['x'] / 1000.0  # mm -> m
                y_m = tgt['y'] / 1000.0
                r_precise = np.hypot(x_m, y_m)  # true range from x, y
                angle_deg = np.degrees(np.arctan2(x_m, y_m))  # 0° = forward

                st.r_m = r_precise  # use precise range for telemetry
                st.angle_deg = angle_deg  # recomputed from x,y too
                st.speed_m_s = tgt['v']
                st.last_seen = now

                # Plot directly in (x, y) — no need to project from polar
                st.trail.append((x_m, y_m))


# ================================= Plot =====================================

class RadarPlot:
    """Cartesian plot in (lateral, range) coordinates with radar styling."""

    def __init__(self, targets: dict, stats: dict, lock: threading.Lock):
        self.targets = targets
        self.stats = stats
        self.lock = lock

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(9, 7))
        self.fig.canvas.manager.set_window_title('RD-03D Radar Viewer')
        self._draw_static()

        # Dynamic artists per target (created lazily)
        self.target_artists = {}   # id -> dict(blip, trail_line, label)

        # Telemetry text
        self.tele_text = self.ax.text(
            0.02, 0.97, '', transform=self.ax.transAxes,
            color='#a0ffa0', fontsize=10, family='monospace',
            verticalalignment='top'
        )

    def _draw_static(self) -> None:
        ax = self.ax
        ax.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
        ax.set_ylim(0, MAX_RANGE_M + 0.2)
        ax.set_aspect('equal')
        ax.set_facecolor('#000000')
        ax.set_xlabel('Поперечная ось X, м', color='#80ff80')
        ax.set_ylabel('Дальность Y, м',       color='#80ff80')
        ax.tick_params(colors='#80ff80')
        for spine in ax.spines.values():
            spine.set_color('#206020')

        # Range rings (semi-circles, since radar covers front half-plane only)
        theta = np.linspace(0, np.pi, 200)
        for r in np.arange(RING_STEP_M, MAX_RANGE_M + 0.001, RING_STEP_M):
            ax.plot(r * np.cos(theta), r * np.sin(theta),
                    color='#208020', linewidth=0.8, alpha=0.7)
            ax.text(0, r + 0.05, f'{r:.0f} м',
                    color='#80ff80', fontsize=8, ha='center')

        # Azimuth lines: -60°, -30°, 0°, +30°, +60° (RD-03D field of view ±60°)
        for ang_deg in (-60, -30, 0, 30, 60):
            rad = np.radians(ang_deg)
            x2 = MAX_RANGE_M * np.sin(rad)
            y2 = MAX_RANGE_M * np.cos(rad)
            ax.plot([0, x2], [0, y2], color='#206020', linewidth=0.6, alpha=0.7)
            # Angle label at the edge
            ax.text(x2 * 1.05, y2 * 1.05, f'{ang_deg:+d}°',
                    color='#80ff80', fontsize=8, ha='center')

        # Radar origin
        ax.plot(0, 0, marker='^', color='#80ff80', markersize=12)
        ax.text(0, -0.15, 'RD-03D', color='#80ff80',
                fontsize=9, ha='center', va='top')

    def _ensure_artist(self, tid: int):
        """Create plot artists for target id on first appearance."""
        if tid in self.target_artists:
            return self.target_artists[tid]

        color = COLORS[(tid - 1) % len(COLORS)]
        blip,      = self.ax.plot([], [], 'o', color=color,
                                  markersize=14, markeredgecolor='white',
                                  markeredgewidth=1.0)
        trail,     = self.ax.plot([], [], '-', color=color,
                                  linewidth=1.5, alpha=0.6)
        label = self.ax.text(0, 0, '', color=color, fontsize=9,
                             fontweight='bold', ha='left', va='bottom')

        self.target_artists[tid] = {
            'blip': blip, 'trail': trail, 'label': label
        }
        return self.target_artists[tid]

    def update(self) -> None:
        """Redraw all dynamic elements."""
        now = time.time()

        with self.lock:
            tele_lines = [f"Frame #{self.stats.get('last_n', 0)}  "
                          f"t={self.stats.get('last_ms', 0)} ms  "
                          f"rx={self.stats.get('frames', 0)}"]

            visible_count = 0
            for tid, st in self.targets.items():
                artist = self._ensure_artist(tid)

                age = now - st.last_seen
                if age > TARGET_TIMEOUT_S:
                    # Target stale -> hide
                    artist['blip'].set_data([], [])
                    artist['trail'].set_data([], [])
                    artist['label'].set_text('')
                    if len(st.trail) > 0:
                        st.trail.clear()
                    continue

                visible_count += 1

                # Latest position
                if len(st.trail) == 0:
                    continue
                tx, ty = st.trail[-1]

                artist['blip'].set_data([tx], [ty])

                trail_xy = np.array(st.trail)
                artist['trail'].set_data(trail_xy[:, 0], trail_xy[:, 1])

                artist['label'].set_position((tx + 0.08, ty + 0.08))
                artist['label'].set_text(
                    f"T{tid}\n{st.r_m:.2f} м\n{st.angle_deg:+.1f}°\n{st.speed_m_s:+.2f} м/с"
                )

                tele_lines.append(
                    f"T{tid}: r={st.r_m:5.2f} м  "
                    f"a={st.angle_deg:+6.1f}°  "
                    f"v={st.speed_m_s:+5.2f} м/с"
                )

            if visible_count == 0:
                tele_lines.append("целей не обнаружено")

            self.tele_text.set_text('\n'.join(tele_lines))


# ================================ Main ======================================

def auto_detect_port() -> str:
    """Try to find an ESP32 USB-UART port automatically."""
    candidates = []
    for p in serial.tools.list_ports.comports():
        descr = (p.description or '').lower()
        if any(k in descr for k in ('cp210', 'ch340', 'ch9102', 'silicon labs', 'usb-serial')):
            candidates.append(p.device)
    if candidates:
        return candidates[0]
    # Fallback: first available port
    ports = list(serial.tools.list_ports.comports())
    if ports:
        return ports[0].device
    raise RuntimeError("No serial ports found. Pass port name as argument.")


def main() -> None:
    ap = argparse.ArgumentParser(description='RD-03D multi-target radar visualizer')
    ap.add_argument('port', nargs='?', default=None,
                    help='Serial port (e.g. COM7 or /dev/ttyUSB0). Auto-detected if omitted.')
    ap.add_argument('--baud', type=int, default=DEFAULT_BAUD,
                    help=f'Baud rate (default {DEFAULT_BAUD})')
    args = ap.parse_args()

    port = args.port or auto_detect_port()

    # Shared state
    lock = threading.Lock()
    targets: dict = {}
    stats: dict = {'frames': 0, 'last_n': 0, 'last_ms': 0}

    # Start background reader
    reader = SerialReader(port, args.baud, lock, targets, stats)
    reader.start()

    # Plot
    radar = RadarPlot(targets, stats, lock)

    # Use a timer instead of FuncAnimation for more predictable behaviour
    def on_timer():
        radar.update()
        radar.fig.canvas.draw_idle()

    timer = radar.fig.canvas.new_timer(interval=ANIM_INTERVAL_MS)
    timer.add_callback(on_timer)
    timer.start()

    try:
        plt.show()
    finally:
        reader.stop_flag = True
        reader.join(timeout=1.0)


if __name__ == '__main__':
    main()
