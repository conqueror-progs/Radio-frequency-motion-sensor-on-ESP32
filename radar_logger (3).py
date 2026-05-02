"""
=============================================================================
  RD-03D multi-target radar logger + visualizer
  -------------------------------------------------------------
  Bachelor's thesis project: RF motion sensor based on ESP32 + RD-03D

  Adds to radar_viz.py:
    - CSV logger with configurable session metadata
    - Hotkeys: SPACE = start/stop recording, M = mark event, Q = quit
    - Auto-filename with timestamp
    - Session summary on stop (frames logged, duration, per-target stats)

  CSV columns:
      session_id, wall_time, frame_n, elapsed_ms, exp_range_m, exp_angle_deg,
      target_id, r_m, x_m, y_m, angle_deg, speed_m_s, event_marker

  Requirements:
      pip install pyserial matplotlib numpy

  Usage:
      python radar_logger.py
      python radar_logger.py COM7
      python radar_logger.py --port COM7 --baud 115200 --range 2.0 --angle 0

  During run:
      SPACE  - start / stop recording
      M      - insert event marker in current log row
      Q      - quit

  Output:
      radar_YYYYMMDD_HHMMSS.csv   (in current directory)
=============================================================================
"""

import argparse
import csv
import datetime
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import serial
import serial.tools.list_ports


# ============================== Configuration ================================

DEFAULT_BAUD      = 115200
MAX_RANGE_M       = 5.0
MAX_TARGETS       = 3
TRAIL_LEN         = 25
TARGET_TIMEOUT_S  = 0.5
COLORS            = ['#ff3030', '#30c0ff', '#ffd030']
RING_STEP_M       = 1.0
ANIM_INTERVAL_MS  = 50

CSV_COLUMNS = [
    'session_id',     # UUID of current recording session
    'wall_time',      # ISO-8601 wall clock
    'frame_n',        # monotonic frame counter from radar
    'elapsed_ms',     # ms since recording started
    'exp_range_m',    # expected (ground truth) range in metres
    'exp_angle_deg',  # expected (ground truth) azimuth in degrees
    'target_id',      # 1 / 2 / 3
    'r_m',            # measured range from (x,y) in metres
    'x_m',            # lateral coordinate, metres
    'y_m',            # range coordinate, metres
    'angle_deg',      # measured azimuth, degrees
    'speed_m_s',      # radial speed, m/s
    'event_marker',   # '' normally, 'MARK' when M key pressed
]


# ============================ Data structures ================================

@dataclass
class TargetState:
    id: int
    r_m: float       = 0.0
    angle_deg: float = 0.0
    speed_m_s: float = 0.0
    x_m: float       = 0.0
    y_m: float       = 0.0
    last_seen: float = 0.0
    trail: deque     = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))


@dataclass
class LoggerState:
    """Mutable recording state, protected by lock from RadarPlot."""
    recording:    bool       = False
    session_id:   str        = ''
    start_time:   float      = 0.0
    rows_written: int        = 0
    writer:       object     = None   # csv.DictWriter | None
    file_handle:  object     = None   # file | None
    filepath:     str        = ''
    next_marker:  bool       = False  # set True when M key pressed
    exp_range_m:  float      = 0.0
    exp_angle_deg: float     = 0.0


# ============================ Serial reader ==================================

class SerialReader(threading.Thread):
    def __init__(self, port, baud, lock, targets, stats, logger_state):
        super().__init__(daemon=True)
        self.port          = port
        self.baud          = baud
        self.lock          = lock
        self.targets       = targets
        self.stats         = stats
        self.logger_state  = logger_state
        self.stop_flag     = False

    def run(self):
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
                print(f"[ERROR] {e}", file=sys.stderr)
                break

            if not line or not line.startswith('{'):
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue

            self._update(frame)

        ser.close()

    def _update(self, frame: dict):
        now = time.time()
        with self.lock:
            self.stats['frames']  += 1
            self.stats['last_n']   = frame.get('n', 0)
            self.stats['last_ms']  = frame.get('t', 0)

            ls = self.logger_state
            mark = ''
            if ls.next_marker:
                mark = 'MARK'
                ls.next_marker = False

            targets_this_frame = []

            for tgt in frame.get('targets', []):
                tid = tgt['id']
                st  = self.targets.setdefault(tid, TargetState(id=tid))

                x_m = tgt['x'] / 1000.0
                y_m = tgt['y'] / 1000.0
                r   = np.hypot(x_m, y_m)
                a   = np.degrees(np.arctan2(x_m, y_m))

                st.r_m       = r
                st.angle_deg = a
                st.speed_m_s = tgt['v']
                st.x_m       = x_m
                st.y_m       = y_m
                st.last_seen = now
                st.trail.append((x_m, y_m))

                targets_this_frame.append(st)

            # Write to CSV if recording
            if ls.recording and ls.writer is not None:
                wall  = datetime.datetime.now().isoformat(timespec='milliseconds')
                ems   = int((now - ls.start_time) * 1000)
                fn    = frame.get('n', 0)

                if targets_this_frame:
                    for st in targets_this_frame:
                        row = {
                            'session_id':    ls.session_id,
                            'wall_time':     wall,
                            'frame_n':       fn,
                            'elapsed_ms':    ems,
                            'exp_range_m':   ls.exp_range_m,
                            'exp_angle_deg': ls.exp_angle_deg,
                            'target_id':     st.id,
                            'r_m':           round(st.r_m, 4),
                            'x_m':           round(st.x_m, 4),
                            'y_m':           round(st.y_m, 4),
                            'angle_deg':     round(st.angle_deg, 3),
                            'speed_m_s':     round(st.speed_m_s, 3),
                            'event_marker':  mark,
                        }
                        ls.writer.writerow(row)
                        ls.rows_written += 1
                        mark = ''   # only first row in frame gets the marker
                else:
                    # Log empty frame so gaps are visible in analysis
                    row = {
                        'session_id':    ls.session_id,
                        'wall_time':     wall,
                        'frame_n':       fn,
                        'elapsed_ms':    ems,
                        'exp_range_m':   ls.exp_range_m,
                        'exp_angle_deg': ls.exp_angle_deg,
                        'target_id':     '',
                        'r_m':           '',
                        'x_m':           '',
                        'y_m':           '',
                        'angle_deg':     '',
                        'speed_m_s':     '',
                        'event_marker':  mark,
                    }
                    ls.writer.writerow(row)
                    ls.rows_written += 1

                ls.file_handle.flush()


# ============================ Radar plot ====================================

class RadarPlot:
    def __init__(self, targets, stats, lock, logger_state, args):
        self.targets      = targets
        self.stats        = stats
        self.lock         = lock
        self.ls           = logger_state
        self.args         = args

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('RD-03D Radar Logger')

        self._draw_static()
        self.target_artists = {}

        # Status bar at bottom
        self.status_text = self.fig.text(
            0.01, 0.01,
            'SPACE = start recording  |  M = mark event  |  Q = quit',
            color='#808080', fontsize=9, family='monospace'
        )
        # Telemetry top-left
        self.tele_text = self.ax.text(
            0.01, 0.99, '', transform=self.ax.transAxes,
            color='#a0ffa0', fontsize=9, family='monospace',
            verticalalignment='top'
        )
        # REC indicator top-right
        self.rec_text = self.ax.text(
            0.99, 0.99, '', transform=self.ax.transAxes,
            color='#ff3030', fontsize=13, fontweight='bold',
            family='monospace', ha='right', verticalalignment='top'
        )

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    # ------------------------------------------------------------------ static
    def _draw_static(self):
        ax = self.ax
        ax.set_xlim(-MAX_RANGE_M, MAX_RANGE_M)
        ax.set_ylim(0, MAX_RANGE_M + 0.3)
        ax.set_aspect('equal')
        ax.set_facecolor('#050505')
        ax.set_xlabel('Поперечная ось X, м', color='#80ff80')
        ax.set_ylabel('Дальность Y, м',       color='#80ff80')
        ax.tick_params(colors='#80ff80')
        for spine in ax.spines.values():
            spine.set_color('#1a4a1a')

        theta = np.linspace(0, np.pi, 300)
        for r in np.arange(RING_STEP_M, MAX_RANGE_M + 0.01, RING_STEP_M):
            ax.plot(r * np.sin(theta), r * np.cos(theta),
                    color='#1a6a1a', linewidth=0.8, alpha=0.7)
            ax.text(0.04, r, f'{r:.0f} м',
                    color='#80ff80', fontsize=8, ha='left', va='bottom')

        for ang in (-60, -30, 0, 30, 60):
            rad = np.radians(ang)
            ax.plot([0, MAX_RANGE_M * np.sin(rad)],
                    [0, MAX_RANGE_M * np.cos(rad)],
                    color='#1a5a1a', linewidth=0.7, alpha=0.7)
            ax.text(MAX_RANGE_M * np.sin(rad) * 1.04,
                    MAX_RANGE_M * np.cos(rad) * 1.02,
                    f'{ang:+d}°', color='#80ff80', fontsize=8, ha='center')

        ax.plot(0, 0, marker='^', color='#80ff80', markersize=12, zorder=10)
        ax.text(0, -0.18, 'RD-03D', color='#80ff80', fontsize=9,
                ha='center', va='top')

    # --------------------------------------------------------------- artists
    def _ensure_artist(self, tid):
        if tid in self.target_artists:
            return self.target_artists[tid]
        color = COLORS[(tid - 1) % len(COLORS)]
        blip,  = self.ax.plot([], [], 'o', color=color, markersize=14,
                              markeredgecolor='white', markeredgewidth=1.0,
                              zorder=5)
        trail, = self.ax.plot([], [], '-', color=color,
                              linewidth=1.8, alpha=0.55, zorder=4)
        label  = self.ax.text(0, 0, '', color=color, fontsize=9,
                              fontweight='bold', ha='left', va='bottom',
                              zorder=6)
        self.target_artists[tid] = {'blip': blip, 'trail': trail, 'label': label}
        return self.target_artists[tid]

    # ------------------------------------------------------------- key events
    def _on_key(self, event):
        # event.key may be None (e.g. modifier-only press)
        key = event.key if event.key else ''
        # Normalise letter keys; space is just ' '
        if len(key) == 1:
            key = key.lower()

        if key == ' ':
            self._toggle_recording()
        elif key == 'm':
            with self.lock:
                if self.ls.recording:
                    self.ls.next_marker = True
                    print('[MARK] Event marker queued')
        elif key == 'q':
            # _toggle_recording itself acquires the lock and stops cleanly
            if self.ls.recording:
                self._toggle_recording()
            plt.close('all')

    def _toggle_recording(self):
        with self.lock:
            if self.ls.recording:
                self._stop_recording()
            else:
                self._start_recording()

    def _start_recording(self):
        ls = self.ls
        ls.session_id    = str(uuid.uuid4())[:8]
        ls.start_time    = time.time()
        ls.rows_written  = 0
        ls.exp_range_m   = self.args.range
        ls.exp_angle_deg = self.args.angle

        # Save next to the script itself, not in CWD.
        # If frozen (pyinstaller) — use directory of the executable.
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).resolve().parent
        else:
            base_dir = Path(__file__).resolve().parent
        logs_dir = base_dir / 'logs'

        try:
            logs_dir.mkdir(exist_ok=True)
        except OSError as e:
            print(f'[ERROR] Cannot create logs directory {logs_dir}: {e}',
                  file=sys.stderr)
            return

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        ls.filepath = str(logs_dir / f'radar_{ts}.csv')

        try:
            ls.file_handle = open(ls.filepath, 'w', newline='', encoding='utf-8')
        except OSError as e:
            print(f'[ERROR] Cannot open log file {ls.filepath}: {e}',
                  file=sys.stderr)
            ls.filepath = ''
            return

        ls.writer = csv.DictWriter(ls.file_handle, fieldnames=CSV_COLUMNS)
        ls.writer.writeheader()
        ls.recording = True

        print(f'\n[REC ▶] Recording started → {ls.filepath}')
        print(f'        Session: {ls.session_id}')
        print(f'        Expected range:  {ls.exp_range_m} m')
        print(f'        Expected angle:  {ls.exp_angle_deg}°')
        print(f'        Press SPACE to stop, M to mark event.\n')

    def _stop_recording(self):
        ls = self.ls
        if not ls.recording:
            return

        ls.recording = False
        duration = time.time() - ls.start_time

        if ls.file_handle is not None:
            try:
                ls.file_handle.close()
            except OSError as e:
                print(f'[WARN] Error closing log file: {e}', file=sys.stderr)
            ls.file_handle = None
            ls.writer = None

        size_kb = 0.0
        if ls.filepath:
            try:
                size_kb = Path(ls.filepath).stat().st_size / 1024
            except OSError:
                pass

        print(f'\n[REC ■] Recording stopped.')
        print(f'        File:     {ls.filepath}  ({size_kb:.1f} KB)')
        print(f'        Duration: {duration:.1f} s')
        print(f'        Rows:     {ls.rows_written}\n')

    # ----------------------------------------------------------------- update
    def update(self):
        now = time.time()
        with self.lock:
            ls   = self.ls
            fn   = self.stats.get('last_n', 0)
            t_ms = self.stats.get('last_ms', 0)
            rx   = self.stats.get('frames', 0)

            rec_label = ''
            if ls.recording:
                elapsed = now - ls.start_time
                rec_label = (f'● REC  {elapsed:.0f}s  '
                             f'{ls.rows_written} rows')

            tele = [f'Frame #{fn}  t={t_ms} ms  rx={rx}']
            if ls.recording:
                tele.append(f'Expected: r={ls.exp_range_m} m  '
                            f'a={ls.exp_angle_deg}°')

            visible = 0
            for tid, st in self.targets.items():
                art = self._ensure_artist(tid)
                age = now - st.last_seen

                if age > TARGET_TIMEOUT_S:
                    art['blip'].set_data([], [])
                    art['trail'].set_data([], [])
                    art['label'].set_text('')
                    st.trail.clear()
                    continue

                visible += 1
                if len(st.trail) == 0:
                    continue

                tx, ty = st.trail[-1]
                art['blip'].set_data([tx], [ty])

                tr = np.array(st.trail)
                art['trail'].set_data(tr[:, 0], tr[:, 1])

                art['label'].set_position((tx + 0.08, ty + 0.06))
                art['label'].set_text(
                    f'T{tid}\n'
                    f'{st.r_m:.3f} м\n'
                    f'{st.angle_deg:+.2f}°\n'
                    f'{st.speed_m_s:+.3f} м/с'
                )

                tele.append(
                    f'T{tid}: r={st.r_m:.3f} м  '
                    f'a={st.angle_deg:+.2f}°  '
                    f'v={st.speed_m_s:+.3f} м/с  '
                    f'x={st.x_m:.3f}  y={st.y_m:.3f}'
                )

            if visible == 0:
                tele.append('целей не обнаружено')

            self.tele_text.set_text('\n'.join(tele))
            self.rec_text.set_text(rec_label)


# =============================== Helpers ====================================

def auto_detect_port():
    for p in serial.tools.list_ports.comports():
        d = (p.description or '').lower()
        if any(k in d for k in ('cp210', 'ch340', 'ch9102', 'silicon labs', 'usb-serial')):
            return p.device
    ports = list(serial.tools.list_ports.comports())
    if ports:
        return ports[0].device
    raise RuntimeError('No serial ports found. Pass port name as argument.')


# ================================= Main =====================================

def main():
    ap = argparse.ArgumentParser(description='RD-03D radar logger + visualizer')
    ap.add_argument('port', nargs='?', default=None,
                    help='Serial port (e.g. COM7). Auto-detected if omitted.')
    ap.add_argument('--baud',  type=int,   default=DEFAULT_BAUD)
    ap.add_argument('--range', type=float, default=0.0,
                    help='Ground-truth target range in metres (for CSV metadata)')
    ap.add_argument('--angle', type=float, default=0.0,
                    help='Ground-truth azimuth in degrees (for CSV metadata)')
    args = ap.parse_args()

    port = args.port or auto_detect_port()

    lock          = threading.Lock()
    targets: dict = {}
    stats: dict   = {'frames': 0, 'last_n': 0, 'last_ms': 0}
    logger_state  = LoggerState(exp_range_m=args.range,
                                exp_angle_deg=args.angle)

    reader = SerialReader(port, args.baud, lock, targets, stats, logger_state)
    reader.start()

    radar = RadarPlot(targets, stats, lock, logger_state, args)

    def on_timer():
        radar.update()
        radar.fig.canvas.draw_idle()

    timer = radar.fig.canvas.new_timer(interval=ANIM_INTERVAL_MS)
    timer.add_callback(on_timer)
    timer.start()

    print('=' * 60)
    print('  RD-03D Radar Logger')
    print(f'  Port : {port} @ {args.baud} bps')
    print(f'  Expected range : {args.range} m')
    print(f'  Expected angle : {args.angle}°')
    print('  SPACE = start/stop recording')
    print('  M     = insert event marker')
    print('  Q     = quit')
    print('=' * 60)

    try:
        plt.show()
    finally:
        # Stop recording cleanly. _toggle_recording handles the lock internally.
        if logger_state.recording:
            radar._toggle_recording()
        reader.stop_flag = True
        reader.join(timeout=1.0)


if __name__ == '__main__':
    main()
