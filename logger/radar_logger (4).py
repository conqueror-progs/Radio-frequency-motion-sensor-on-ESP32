"""
=============================================================================
  RD-03D multi-target radar logger + visualizer
  -------------------------------------------------------------
  Bachelor's thesis project: RF motion sensor based on ESP32 + RD-03D

  Features:
    - Real-time visualisation in radar polar coordinates
    - GNN tracker with persistent track IDs (P1, P2, ...)
    - Anti-phantom filter
    - CSV logger with experiment scenario tagging
    - Scenarios loaded from experiments.json, switchable on the fly

  CSV columns:
      session_id, wall_time, frame_n, elapsed_ms,
      scenario_id, scenario_desc, exp_n_targets,
      exp_obstacle, exp_obstacle_mm, exp_range_m, exp_angle_deg,
      target_id, r_m, x_m, y_m, angle_deg, speed_m_s,
      event_marker

  Requirements:
      pip install pyserial matplotlib numpy

  Usage:
      python radar_logger.py
      python radar_logger.py COM7
      python radar_logger.py --port COM7 --baud 115200
      python radar_logger.py --scenarios my_experiments.json

  Hotkeys during run:
      SPACE  - start / stop recording
      1..9   - switch to scenario with this key (per experiments.json)
      N      - clear scenario ('no scenario' / pristrelka mode)
      M      - insert event marker in current log row
      Q      - quit

  Output:
      <script_dir>/logs/radar_YYYYMMDD_HHMMSS.csv
=============================================================================
"""

import argparse
import csv
import datetime
import json
import math
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

# ---- Anti-phantom filter ---------------------------------------------------
# RD-03D occasionally keeps a target slot 'frozen' at the last known position
# after the real subject has left the FOV. We detect this as N consecutive
# frames in the same XY position (within JITTER_M) and discard such targets.
PHANTOM_FRAMES    = 8       # frames of zero motion to mark as phantom
PHANTOM_JITTER_M  = 0.005   # 5 mm — anything below this is "no movement"

# ---- Persistent-track GNN tracker ------------------------------------------
# We don't trust radar's slot IDs (1/2/3) — they get reassigned when a target
# leaves the FOV. Instead we run a Global Nearest-Neighbour associator on top
# of the raw slot stream and keep our own stable track IDs (P1, P2, ...).
MAX_ASSOC_DIST_M    = 0.60  # max distance to associate a slot with a track
MIN_HITS_FOR_CONFIRM = 3    # slot detections required to confirm a new track
MAX_COAST_FRAMES    = 6     # frames a confirmed track is kept after losing slot
MAX_TENTATIVE_AGE_S = 0.5   # tentative tracks expire faster than confirmed

CSV_COLUMNS = [
    'session_id',       # UUID of current recording session
    'wall_time',        # ISO-8601 wall clock
    'frame_n',          # monotonic frame counter from radar
    'elapsed_ms',       # ms since recording started
    # ---- Scenario tagging (from experiments.json) -------------------------
    'scenario_id',      # short id, e.g. 'static_2m_normal'
    'scenario_desc',    # human-readable description
    'exp_n_targets',    # expected number of targets (for FP/FN calculation)
    'exp_obstacle',     # 'none' | 'drywall' | 'wood' | 'concrete' | ...
    'exp_obstacle_mm',  # obstacle thickness in mm
    'exp_range_m',      # expected (ground truth) range in metres
    'exp_angle_deg',    # expected (ground truth) azimuth in degrees
    # ---- Target measurement ------------------------------------------------
    'target_id',        # P1, P2, P3, ...
    'r_m',              # measured range from (x,y) in metres
    'x_m',              # lateral coordinate, metres
    'y_m',              # range coordinate, metres
    'angle_deg',        # measured azimuth, degrees
    'speed_m_s',        # radial speed, m/s
    # ---- Annotations -------------------------------------------------------
    'event_marker',     # '' normally, 'MARK' when M key pressed,
                        # 'SCENARIO_START' on scenario switch
]


# ============================ Data structures ================================

@dataclass
class SlotState:
    """
    Raw radar slot (id 1/2/3 from JSON).
    Used internally for phantom detection BEFORE association.
    """
    static_count: int  = 0
    is_phantom:   bool = False
    last_x_m:     float = 0.0
    last_y_m:     float = 0.0


@dataclass
class Track:
    """
    Persistent track with stable ID. Survives radar slot reassignment.
    Created via GNN association of incoming radar measurements.
    """
    track_id:    int
    x_m:         float = 0.0
    y_m:         float = 0.0
    r_m:         float = 0.0
    angle_deg:   float = 0.0
    speed_m_s:   float = 0.0
    # Velocity estimate for prediction during coasting
    vx_m_s:      float = 0.0
    vy_m_s:      float = 0.0
    last_seen:   float = 0.0
    last_update_t: float = 0.0       # time.time() of last association
    hits:        int   = 0           # successful associations
    misses:      int   = 0           # consecutive frames without association
    confirmed:   bool  = False       # true after MIN_HITS_FOR_CONFIRM hits
    trail: deque = field(default_factory=lambda: deque(maxlen=TRAIL_LEN))


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
    next_scenario_marker: bool = False  # set when scenario is switched
    book:         object     = None   # ScenarioBook reference


# ============================ Experiment scenarios ==========================

@dataclass
class Scenario:
    """One experiment scenario loaded from experiments.json."""
    key:                   str          # hotkey character: '1'..'9'
    id:                    str          # short id, e.g. 'static_2m_normal'
    description:           str
    n_targets:             int   = 1
    range_m:               float = 0.0
    angle_deg:             float = 0.0
    obstacle:              str   = 'none'
    obstacle_thickness_mm: int   = 0


class ScenarioBook:
    """
    Holds the full set of scenarios from experiments.json plus the currently
    selected one. A 'NONE' scenario is always available — represents the
    'no scenario' state for between-experiment pristrelka.
    """
    NONE = Scenario(key='', id='', description='(no scenario)')

    def __init__(self):
        self.by_key:  dict[str, Scenario] = {}
        self.ordered: list[Scenario]      = []
        self.current: Scenario            = ScenarioBook.NONE

    def load(self, path: Path) -> None:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f'[WARN] Cannot load scenarios from {path}: {e}',
                  file=sys.stderr)
            return

        for item in data.get('scenarios', []):
            try:
                sc = Scenario(
                    key=str(item['key']).strip(),
                    id=str(item['id']),
                    description=str(item.get('description', '')),
                    n_targets=int(item.get('n_targets', 1)),
                    range_m=float(item.get('range_m', 0.0)),
                    angle_deg=float(item.get('angle_deg', 0.0)),
                    obstacle=str(item.get('obstacle', 'none')),
                    obstacle_thickness_mm=int(item.get('obstacle_thickness_mm', 0)),
                )
            except (KeyError, ValueError, TypeError) as e:
                print(f'[WARN] Skipping malformed scenario {item}: {e}',
                      file=sys.stderr)
                continue

            if sc.key in self.by_key:
                print(f'[WARN] Duplicate hotkey "{sc.key}" — '
                      f'scenario "{sc.id}" overrides previous.', file=sys.stderr)
            self.by_key[sc.key] = sc
            self.ordered.append(sc)

        print(f'[INFO] Loaded {len(self.ordered)} scenarios from {path.name}')

    def select_by_key(self, key: str) -> bool:
        """Switch current scenario by hotkey. Return True if changed."""
        if key in self.by_key and self.current is not self.by_key[key]:
            self.current = self.by_key[key]
            return True
        return False

    def clear_current(self) -> bool:
        """Switch back to NONE state. Return True if was non-NONE."""
        if self.current is not ScenarioBook.NONE:
            self.current = ScenarioBook.NONE
            return True
        return False


# =========================== GNN tracker ====================================

class TrackerEngine:
    """
    Global Nearest Neighbour associator + track manager.
    All public methods must be called with the lock held by the caller.
    """
    def __init__(self):
        self.slots:  dict[int, SlotState] = {}    # raw radar slot id -> state
        self.tracks: dict[int, Track]     = {}    # our persistent ID -> Track
        self._next_track_id: int          = 1

    # ------------------------------------------------------------------ public
    def process(self, raw_targets: list, now: float) -> list:
        """
        Take a list of dicts {id, x, y, v} from the radar (slot space),
        run phantom filter and GNN association, return list of confirmed Tracks.
        """
        # 1) Phantom filtering on slot space
        live_measurements = []  # list of (x_m, y_m, speed_m_s)
        for tgt in raw_targets:
            sid = tgt['id']
            x_m = tgt['x'] / 1000.0
            y_m = tgt['y'] / 1000.0
            v   = float(tgt.get('v', 0.0))

            slot = self.slots.setdefault(sid, SlotState())
            moved = math.hypot(x_m - slot.last_x_m, y_m - slot.last_y_m)
            if moved < PHANTOM_JITTER_M:
                slot.static_count += 1
                if slot.static_count >= PHANTOM_FRAMES:
                    slot.is_phantom = True
            else:
                slot.static_count = 0
                slot.is_phantom   = False
            slot.last_x_m, slot.last_y_m = x_m, y_m

            if not slot.is_phantom:
                live_measurements.append((x_m, y_m, v))

        # 2) GNN association
        self._associate(live_measurements, now)

        # 3) Track lifecycle: confirm / drop
        self._lifecycle(now)

        # 4) Return only confirmed tracks for display & logging
        return [t for t in self.tracks.values() if t.confirmed]

    # ----------------------------------------------------------------- private
    def _associate(self, measurements: list, now: float) -> None:
        """
        Greedy GNN: build all (track, measurement) pairs sorted by distance,
        assign each in order, skipping tracks/measurements already assigned.
        Predicts position of each existing track for the time elapsed since
        last update — better matching for moving targets.
        """
        # Predict track positions for current time
        predicted = {}
        for tid, tr in self.tracks.items():
            dt = now - tr.last_update_t if tr.last_update_t > 0 else 0.0
            # Cap dt to avoid runaway prediction during long stalls
            dt = min(dt, 0.3)
            predicted[tid] = (tr.x_m + tr.vx_m_s * dt,
                              tr.y_m + tr.vy_m_s * dt)

        # Build all pairs (dist, track_id, meas_idx)
        pairs = []
        for tid, (px, py) in predicted.items():
            for mi, (mx, my, _) in enumerate(measurements):
                d = math.hypot(mx - px, my - py)
                if d <= MAX_ASSOC_DIST_M:
                    pairs.append((d, tid, mi))

        pairs.sort()  # ascending distance
        used_tracks: set = set()
        used_meas:   set = set()

        for d, tid, mi in pairs:
            if tid in used_tracks or mi in used_meas:
                continue
            used_tracks.add(tid)
            used_meas.add(mi)
            self._update_track(self.tracks[tid], measurements[mi], now)

        # Tracks that didn't get a measurement: increment miss counter
        for tid, tr in self.tracks.items():
            if tid not in used_tracks:
                tr.misses += 1

        # Measurements that weren't matched: spawn new tentative tracks
        for mi, m in enumerate(measurements):
            if mi in used_meas:
                continue
            self._spawn_track(m, now)

    def _update_track(self, tr: Track, m: tuple, now: float) -> None:
        x_m, y_m, v = m
        # Velocity estimate via simple alpha-beta filter
        dt = now - tr.last_update_t if tr.last_update_t > 0 else 0.1
        if dt > 1e-3:
            vx_meas = (x_m - tr.x_m) / dt
            vy_meas = (y_m - tr.y_m) / dt
            alpha = 0.5
            tr.vx_m_s = (1 - alpha) * tr.vx_m_s + alpha * vx_meas
            tr.vy_m_s = (1 - alpha) * tr.vy_m_s + alpha * vy_meas

        tr.x_m       = x_m
        tr.y_m       = y_m
        tr.r_m       = math.hypot(x_m, y_m)
        tr.angle_deg = math.degrees(math.atan2(x_m, y_m))
        tr.speed_m_s = v
        tr.last_seen = now
        tr.last_update_t = now
        tr.hits   += 1
        tr.misses  = 0
        tr.trail.append((x_m, y_m))
        if not tr.confirmed and tr.hits >= MIN_HITS_FOR_CONFIRM:
            tr.confirmed = True

    def _spawn_track(self, m: tuple, now: float) -> None:
        x_m, y_m, v = m
        tr = Track(
            track_id=self._next_track_id,
            x_m=x_m, y_m=y_m,
            r_m=math.hypot(x_m, y_m),
            angle_deg=math.degrees(math.atan2(x_m, y_m)),
            speed_m_s=v,
            last_seen=now, last_update_t=now,
            hits=1, misses=0, confirmed=False,
        )
        tr.trail.append((x_m, y_m))
        self.tracks[self._next_track_id] = tr
        self._next_track_id += 1

    def _lifecycle(self, now: float) -> None:
        """Drop tracks that have coasted too long or never confirmed."""
        to_drop = []
        for tid, tr in self.tracks.items():
            age = now - tr.last_update_t
            if tr.confirmed:
                if tr.misses > MAX_COAST_FRAMES:
                    to_drop.append(tid)
            else:
                # tentative track: drop quickly
                if age > MAX_TENTATIVE_AGE_S:
                    to_drop.append(tid)
        for tid in to_drop:
            del self.tracks[tid]


# ============================ Serial reader ==================================

class SerialReader(threading.Thread):
    def __init__(self, port, baud, lock, tracker, stats, logger_state):
        super().__init__(daemon=True)
        self.port          = port
        self.baud          = baud
        self.lock          = lock
        self.tracker       = tracker          # TrackerEngine
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

            # Determine event marker: SCENARIO_START takes precedence over MARK
            mark = ''
            if ls.next_scenario_marker:
                mark = 'SCENARIO_START'
                ls.next_scenario_marker = False
            elif ls.next_marker:
                mark = 'MARK'
                ls.next_marker = False

            # Run phantom filter + GNN association in one shot
            confirmed_tracks = self.tracker.process(
                frame.get('targets', []), now)

            # Build per-frame scenario dict (cached per call to keep CSV consistent)
            sc = ls.book.current if ls.book is not None else ScenarioBook.NONE
            scenario_fields = {
                'scenario_id':     sc.id,
                'scenario_desc':   sc.description,
                'exp_n_targets':   sc.n_targets if sc.id else '',
                'exp_obstacle':    sc.obstacle if sc.id else '',
                'exp_obstacle_mm': sc.obstacle_thickness_mm if sc.id else '',
                'exp_range_m':     sc.range_m if sc.id else '',
                'exp_angle_deg':   sc.angle_deg if sc.id else '',
            }

            # Write to CSV if recording
            if ls.recording and ls.writer is not None:
                wall = datetime.datetime.now().isoformat(timespec='milliseconds')
                ems  = int((now - ls.start_time) * 1000)
                fn   = frame.get('n', 0)

                base_row = {
                    'session_id': ls.session_id,
                    'wall_time':  wall,
                    'frame_n':    fn,
                    'elapsed_ms': ems,
                    **scenario_fields,
                }

                if confirmed_tracks:
                    for tr in confirmed_tracks:
                        row = {
                            **base_row,
                            'target_id':     f'P{tr.track_id}',
                            'r_m':           round(tr.r_m, 4),
                            'x_m':           round(tr.x_m, 4),
                            'y_m':           round(tr.y_m, 4),
                            'angle_deg':     round(tr.angle_deg, 3),
                            'speed_m_s':     round(tr.speed_m_s, 3),
                            'event_marker':  mark,
                        }
                        ls.writer.writerow(row)
                        ls.rows_written += 1
                        mark = ''   # only first row in frame gets the marker
                else:
                    # Log empty frame so gaps are visible in analysis
                    row = {
                        **base_row,
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
    def __init__(self, tracker, stats, lock, logger_state, args):
        self.tracker      = tracker
        self.stats        = stats
        self.lock         = lock
        self.ls           = logger_state
        self.args         = args

        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title('RD-03D Radar Logger')

        self._draw_static()
        self.target_artists = {}   # track_id -> {blip, trail, label}

        # Status bar at bottom
        self.status_text = self.fig.text(
            0.01, 0.01,
            'SPACE=rec  M=mark  1..9=scenario  N=clear scenario  Q=quit',
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

        # Range arcs cover the full FOV including ±60°.
        # Param: phi ∈ [-π/2, +π/2], where x = r·sin(phi), y = r·cos(phi)
        # gives an upper half-circle with phi=0 at the top.
        phi = np.linspace(-np.pi/2, np.pi/2, 300)
        for r in np.arange(RING_STEP_M, MAX_RANGE_M + 0.01, RING_STEP_M):
            ax.plot(r * np.sin(phi), r * np.cos(phi),
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
        elif key in '123456789':
            self._switch_scenario(key)
        elif key == 'n':
            self._clear_scenario()
        elif key == 'q':
            # _toggle_recording itself acquires the lock and stops cleanly
            if self.ls.recording:
                self._toggle_recording()
            plt.close('all')

    def _switch_scenario(self, key: str):
        with self.lock:
            book = self.ls.book
            if book is None:
                return
            if book.select_by_key(key):
                self.ls.next_scenario_marker = True
                sc = book.current
                print(f'[SCENARIO] -> [{sc.key}] {sc.id}: {sc.description}')
            elif key not in book.by_key:
                print(f'[SCENARIO] No scenario bound to key "{key}"')

    def _clear_scenario(self):
        with self.lock:
            book = self.ls.book
            if book is None:
                return
            if book.clear_current():
                self.ls.next_scenario_marker = True
                print('[SCENARIO] -> (no scenario)')

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
        # Mark first frame with SCENARIO_START so scenario boundaries are clear
        ls.next_scenario_marker = True

        sc = ls.book.current if ls.book is not None else ScenarioBook.NONE
        print(f'\n[REC ▶] Recording started → {ls.filepath}')
        print(f'        Session:  {ls.session_id}')
        if sc.id:
            print(f'        Scenario: [{sc.key}] {sc.id} — {sc.description}')
            print(f'                  range={sc.range_m} m  angle={sc.angle_deg}°  '
                  f'n_targets={sc.n_targets}  obstacle={sc.obstacle}')
        else:
            print(f'        Scenario: (none — press 1..9 to select)')
        print(f'        Press SPACE to stop, M to mark event,')
        print(f'        1..9 to switch scenario, N to clear scenario.\n')

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

            # Show current scenario in telemetry
            if ls.book is not None:
                sc = ls.book.current
                if sc.id:
                    tele.append(
                        f'Scenario: [{sc.key}] {sc.id} '
                        f'(r={sc.range_m}m a={sc.angle_deg}° '
                        f'n={sc.n_targets} obstacle={sc.obstacle})'
                    )
                else:
                    tele.append('Scenario: (none — press 1..9)')

            # Snapshot live confirmed tracks
            live_track_ids = set()
            visible = 0
            for tid, tr in self.tracker.tracks.items():
                if not tr.confirmed:
                    continue
                live_track_ids.add(tid)
                art = self._ensure_artist(tid)

                if len(tr.trail) == 0:
                    continue

                tx, ty = tr.trail[-1]
                art['blip'].set_data([tx], [ty])

                trail_xy = np.array(tr.trail)
                art['trail'].set_data(trail_xy[:, 0], trail_xy[:, 1])

                # Visual cue for coasting (predicted, not measured this frame)
                if tr.misses > 0:
                    art['blip'].set_alpha(0.5)
                    art['trail'].set_alpha(0.3)
                else:
                    art['blip'].set_alpha(1.0)
                    art['trail'].set_alpha(0.55)

                art['label'].set_position((tx + 0.08, ty + 0.06))
                art['label'].set_text(
                    f'P{tr.track_id}\n'
                    f'{tr.r_m:.3f} м\n'
                    f'{tr.angle_deg:+.2f}°\n'
                    f'{tr.speed_m_s:+.3f} м/с'
                )

                tele.append(
                    f'P{tr.track_id}: r={tr.r_m:.3f} м  '
                    f'a={tr.angle_deg:+.2f}°  '
                    f'v={tr.speed_m_s:+.3f} м/с  '
                    f'x={tr.x_m:.3f}  y={tr.y_m:.3f}'
                    + (f'  [coast {tr.misses}]' if tr.misses > 0 else '')
                )
                visible += 1

            # Hide and remove artists for tracks that no longer exist
            stale_artist_ids = set(self.target_artists.keys()) - live_track_ids
            for tid in stale_artist_ids:
                art = self.target_artists.pop(tid)
                art['blip'].remove()
                art['trail'].remove()
                art['label'].remove()

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
    ap.add_argument('--baud',      type=int, default=DEFAULT_BAUD)
    ap.add_argument('--scenarios', type=str, default='experiments.json',
                    help='Path to scenarios JSON (default: experiments.json '
                         'next to script)')
    args = ap.parse_args()

    port = args.port or auto_detect_port()

    # Resolve scenarios JSON path: absolute -> as-is; relative -> next to script
    scenarios_path = Path(args.scenarios)
    if not scenarios_path.is_absolute():
        if getattr(sys, 'frozen', False):
            base_dir = Path(sys.executable).resolve().parent
        else:
            base_dir = Path(__file__).resolve().parent
        scenarios_path = base_dir / scenarios_path

    book = ScenarioBook()
    if scenarios_path.exists():
        book.load(scenarios_path)
    else:
        print(f'[WARN] Scenarios file not found: {scenarios_path}')
        print('       Logger will work but scenario keys 1..9 will be inactive.')

    lock          = threading.Lock()
    tracker       = TrackerEngine()
    stats: dict   = {'frames': 0, 'last_n': 0, 'last_ms': 0}
    logger_state  = LoggerState(book=book)

    reader = SerialReader(port, args.baud, lock, tracker, stats, logger_state)
    reader.start()

    radar = RadarPlot(tracker, stats, lock, logger_state, args)

    def on_timer():
        radar.update()
        radar.fig.canvas.draw_idle()

    timer = radar.fig.canvas.new_timer(interval=ANIM_INTERVAL_MS)
    timer.add_callback(on_timer)
    timer.start()

    print('=' * 60)
    print('  RD-03D Radar Logger')
    print(f'  Port      : {port} @ {args.baud} bps')
    print(f'  Scenarios : {scenarios_path.name} '
          f'({len(book.ordered)} loaded)')
    if book.ordered:
        print('              Available hotkeys:')
        for sc in book.ordered:
            print(f'                  [{sc.key}] {sc.id}: {sc.description}')
    print('  Hotkeys   : SPACE=rec  M=mark  1..9=scenario  N=clear  Q=quit')
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
