"""
Assetto Corsa Shared Memory Reader
===================================
Reads from the three AC named memory-mapped files:
  - Local\acpmf_physics   (SPageFilePhysics)
  - Local\acpmf_graphics  (SPageFileGraphic)
  - Local\acpmf_static    (SPageFileStatic)

Struct layout is taken directly from the official C# reference library:
  https://github.com/mdjarv/assettocorsasharedmemory

⚠️  GAP LIMITATION — original AC only:
    AC's shared memory exposes only the PLAYER car. There are no fields for
    other cars' positions, lap times, or gaps (gap_ahead / gap_behind /
    gap_to_leader are all None). This is a hard API limit; it does not exist
    in ACC.  Your existing UDP SPOT packets (parse_rtlap) are the best source
    for per-car lap times if you need relative gaps.

Usage (mirrors your existing udp_client.py pattern):

    from ac_shared_memory import start_shared_memory, stop_shared_memory, get_shared_memory_data

    start_shared_memory()          # call once at app startup

    data = get_shared_memory_data()
    # {
    #   'connected': True,
    #   'flag': 'yellow',           # 'green'|'yellow'|'black'|'white'|
    #                               # 'checkered'|'blue'|'penalty'|'none'
    #   'flag_raw': 2,
    #   'position': 3,
    #   'completed_laps': 4,
    #   'current_lap_ms': 87432,
    #   'last_lap_ms': 92100,
    #   'best_lap_ms': 91500,
    #   'session_time_left': 1234.5,
    #   'normalized_car_position': 0.42,
    #   'is_in_pit': False,
    #   'is_in_pit_lane': False,
    #   'status': 2,                # 0=off 1=replay 2=live 3=paused
    #   'session_type': 2,          # -1=unk 0=practice 1=quali 2=race …
    #   'gap_ahead_ms': None,       # not available in AC shared memory
    #   'gap_behind_ms': None,      # not available in AC shared memory
    #   'gap_to_leader_ms': None,   # not available in AC shared memory
    # }

    stop_shared_memory()
"""

import ctypes
import mmap
import threading
import time
from ctypes import c_int32, c_float, c_wchar
from enum import IntEnum


# ---------------------------------------------------------------------------
# Enums  (from AssettoCorsa.cs / official AC shared memory docs)
# ---------------------------------------------------------------------------

class AC_STATUS(IntEnum):
    AC_OFF    = 0
    AC_REPLAY = 1
    AC_LIVE   = 2
    AC_PAUSE  = 3

class AC_SESSION_TYPE(IntEnum):
    AC_UNKNOWN     = -1
    AC_PRACTICE    = 0
    AC_QUALIFY     = 1
    AC_RACE        = 2
    AC_HOTLAP      = 3
    AC_TIME_ATTACK = 4
    AC_DRIFT       = 5
    AC_DRAG        = 6

class AC_FLAG_TYPE(IntEnum):
    AC_NO_FLAG        = 0
    AC_BLUE_FLAG      = 1
    AC_YELLOW_FLAG    = 2
    AC_BLACK_FLAG     = 3
    AC_WHITE_FLAG     = 4
    AC_CHECKERED_FLAG = 5
    AC_PENALTY_FLAG   = 6

_FLAG_NAMES = {
    AC_FLAG_TYPE.AC_NO_FLAG:        "green",      # no flag shown = green conditions
    AC_FLAG_TYPE.AC_BLUE_FLAG:      "blue",
    AC_FLAG_TYPE.AC_YELLOW_FLAG:    "yellow",
    AC_FLAG_TYPE.AC_BLACK_FLAG:     "black",
    AC_FLAG_TYPE.AC_WHITE_FLAG:     "white",
    AC_FLAG_TYPE.AC_CHECKERED_FLAG: "checkered",
    AC_FLAG_TYPE.AC_PENALTY_FLAG:   "penalty",
}


# ---------------------------------------------------------------------------
# ctypes structs — Pack=4, CharSet=Unicode, matching the C# StructLayout
# exactly as defined in Physics.cs, Graphics.cs, StaticInfo.cs
# ---------------------------------------------------------------------------

class _Coordinates(ctypes.Structure):
    """Used for TyreContact* arrays in Physics."""
    _pack_ = 4
    _fields_ = [
        ('X', c_float),
        ('Y', c_float),
        ('Z', c_float),
    ]


class SPageFilePhysics(ctypes.Structure):
    """
    Physics.cs — Pack=4, CharSet=Unicode
    Fields added per AC version are commented with their version tag.
    """
    _pack_ = 4
    _fields_ = [
        ('PacketId',            c_int32),
        ('Gas',                 c_float),
        ('Brake',               c_float),
        ('Fuel',                c_float),
        ('Gear',                c_int32),
        ('Rpms',                c_int32),
        ('SteerAngle',          c_float),
        ('SpeedKmh',            c_float),
        ('Velocity',            c_float * 3),
        ('AccG',                c_float * 3),
        ('WheelSlip',           c_float * 4),
        ('WheelLoad',           c_float * 4),
        ('WheelsPressure',      c_float * 4),
        ('WheelAngularSpeed',   c_float * 4),
        ('TyreWear',            c_float * 4),
        ('TyreDirtyLevel',      c_float * 4),
        ('TyreCoreTemperature', c_float * 4),
        ('CamberRad',           c_float * 4),
        ('SuspensionTravel',    c_float * 4),
        ('Drs',                 c_float),
        ('TC',                  c_float),
        ('Heading',             c_float),
        ('Pitch',               c_float),
        ('Roll',                c_float),
        ('CgHeight',            c_float),
        ('CarDamage',           c_float * 5),
        ('NumberOfTyresOut',    c_int32),
        ('PitLimiterOn',        c_int32),
        ('Abs',                 c_float),
        ('KersCharge',          c_float),      # since 1.5
        ('KersInput',           c_float),
        ('AutoShifterOn',       c_int32),
        ('RideHeight',          c_float * 2),
        ('TurboBoost',          c_float),      # since 1.5
        ('Ballast',             c_float),
        ('AirDensity',          c_float),
        ('AirTemp',             c_float),      # since 1.6
        ('RoadTemp',            c_float),
        ('LocalAngularVelocity',c_float * 3),
        ('FinalFF',             c_float),
        # since 1.7
        ('PerformanceMeter',    c_float),
        ('EngineBrake',         c_int32),
        ('ErsRecoveryLevel',    c_int32),
        ('ErsPowerLevel',       c_int32),
        ('ErsHeatCharging',     c_int32),
        ('ErsisCharging',       c_int32),
        ('KersCurrentKJ',       c_float),
        ('DrsAvailable',        c_int32),
        ('DrsEnabled',          c_int32),
        ('BrakeTemp',           c_float * 4),
        # since 1.10
        ('Clutch',              c_float),
        ('TyreTempI',           c_float * 4),
        ('TyreTempM',           c_float * 4),
        ('TyreTempO',           c_float * 4),
        # since 1.10.2
        ('IsAIControlled',      c_int32),
        # since 1.11
        ('TyreContactPoint',    _Coordinates * 4),
        ('TyreContactNormal',   _Coordinates * 4),
        ('TyreContactHeading',  _Coordinates * 4),
        ('BrakeBias',           c_float),
        # since 1.12
        ('LocalVelocity',       c_float * 3),
    ]


class SPageFileGraphic(ctypes.Structure):
    """
    Graphics.cs — Pack=4, CharSet=Unicode
    Source: mdjarv/assettocorsasharedmemory  +  official AC shared memory docs.
    """
    _pack_ = 4
    _fields_ = [
        ('PacketId',               c_int32),
        ('Status',                 c_int32),       # AC_STATUS
        ('Session',                c_int32),        # AC_SESSION_TYPE
        ('CurrentTime',            c_wchar * 15),
        ('LastTime',               c_wchar * 15),
        ('BestTime',               c_wchar * 15),
        ('Split',                  c_wchar * 15),
        ('CompletedLaps',          c_int32),
        ('Position',               c_int32),
        ('iCurrentTime',           c_int32),        # current lap ms
        ('iLastTime',              c_int32),        # last lap ms
        ('iBestTime',              c_int32),        # best lap ms
        ('SessionTimeLeft',        c_float),
        ('DistanceTraveled',       c_float),
        ('IsInPit',                c_int32),
        ('CurrentSectorIndex',     c_int32),
        ('LastSectorTime',         c_int32),
        ('NumberOfLaps',           c_int32),
        ('TyreCompound',           c_wchar * 33),
        ('ReplayTimeMultiplier',   c_float),
        ('NormalizedCarPosition',  c_float),
        ('CarCoordinates',         c_float * 3),
        ('PenaltyTime',            c_float),
        ('Flag',                   c_int32),        # AC_FLAG_TYPE
        ('IdealLineOn',            c_int32),
        ('IsInPitLane',            c_int32),
        ('SurfaceGrip',            c_float),
        ('MandatoryPitDone',       c_int32),
        ('WindSpeed',              c_float),
        ('WindDirection',          c_float),
    ]


class SPageFileStatic(ctypes.Structure):
    """
    StaticInfo.cs — Pack=4, CharSet=Unicode
    Source: mdjarv/assettocorsasharedmemory
    """
    _pack_ = 4
    _fields_ = [
        ('SMVersion',                c_wchar * 15),
        ('ACVersion',                c_wchar * 15),
        ('NumberOfSessions',         c_int32),
        ('NumCars',                  c_int32),
        ('CarModel',                 c_wchar * 33),
        ('Track',                    c_wchar * 33),
        ('PlayerName',               c_wchar * 33),
        ('PlayerSurname',            c_wchar * 33),
        ('PlayerNick',               c_wchar * 33),
        ('SectorCount',              c_int32),
        ('MaxTorque',                c_float),
        ('MaxPower',                 c_float),
        ('MaxRpm',                   c_int32),
        ('MaxFuel',                  c_float),
        ('SuspensionMaxTravel',      c_float * 4),
        ('TyreRadius',               c_float * 4),
        # since 1.5
        ('MaxTurboBoost',            c_float),
        ('Deprecated1',              c_float),   # was AirTemp before 1.6
        ('Deprecated2',              c_float),   # was RoadTemp before 1.6
        ('PenaltiesEnabled',         c_int32),
        ('AidFuelRate',              c_float),
        ('AidTireRate',              c_float),
        ('AidMechanicalDamage',      c_float),
        ('AidAllowTyreBlankets',     c_int32),
        ('AidStability',             c_float),
        ('AidAutoClutch',            c_int32),
        ('AidAutoBlip',              c_int32),
        # since 1.7.1
        ('HasDRS',                   c_int32),
        ('HasERS',                   c_int32),
        ('HasKERS',                  c_int32),
        ('KersMaxJoules',            c_float),
        ('EngineBrakeSettingsCount', c_int32),
        ('ErsPowerControllerCount',  c_int32),
        # since 1.7.2
        ('TrackSPlineLength',        c_float),
        ('TrackConfiguration',       c_wchar * 15),
        # since 1.10.2
        ('ErsMaxJ',                  c_float),
        # since 1.13
        ('IsTimedRace',              c_int32),
        ('HasExtraLap',              c_int32),
        ('CarSkin',                  c_wchar * 33),
        ('ReversedGridPositions',    c_int32),
        ('PitWindowStart',           c_int32),
        ('PitWindowEnd',             c_int32),
    ]


# ---------------------------------------------------------------------------
# Reader class
# ---------------------------------------------------------------------------

class AcSharedMemory:
    """
    Opens the three AC named memory-mapped files and polls them on a
    background thread, exposing a thread-safe get_data() snapshot.

    AC uses Windows named MMFs — mmap(0, size, name) opens "Local\\<name>".
    Works on Windows only (AC is Windows-only anyway).
    """

    _MMAP_PHYSICS  = "acpmf_physics"
    _MMAP_GRAPHICS = "acpmf_graphics"
    _MMAP_STATIC   = "acpmf_static"

    def __init__(self, poll_interval: float = 0.05):
        """
        Args:
            poll_interval: Seconds between reads. Default 0.05 s = 20 Hz.
                           Physics in AC updates at ~333 Hz, but 20 Hz is
                           more than enough for flags / position data.
        """
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._mm_physics  = None
        self._mm_graphics = None
        self._mm_static   = None

        self._physics  = None
        self._graphics = None
        self._static   = None

        self._connected = False
        self._data: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Open shared memory pages and start polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[AC SM] Shared memory reader started.")

    def stop(self):
        """Stop polling and close all MMF handles."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close()
        print("[AC SM] Shared memory reader stopped.")

    def get_data(self) -> dict:
        """
        Thread-safe snapshot of the latest values.

        Returns
        -------
        dict with keys:
            connected               bool    True once MMF opened successfully
            flag                    str     'green'|'yellow'|'black'|'white'|
                                            'checkered'|'blue'|'penalty'|'none'
            flag_raw                int     raw AC_FLAG_TYPE int
            position                int     race position (1-based)
            completed_laps          int     laps completed this session
            current_lap_ms          int     current lap time in ms
            last_lap_ms             int     last completed lap time in ms
            best_lap_ms             int     personal best lap time in ms
            session_time_left       float   seconds remaining in session
            normalized_car_position float   0.0–1.0 spline position on track
            is_in_pit               bool    player is in pit box
            is_in_pit_lane          bool    player is in pit lane
            status                  int     AC_STATUS value
            session_type            int     AC_SESSION_TYPE value
            gap_ahead_ms            None    ⚠ not in AC shared memory
            gap_behind_ms           None    ⚠ not in AC shared memory
            gap_to_leader_ms        None    ⚠ not in AC shared memory
        """
        with self._lock:
            return dict(self._data)

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open(self) -> bool:
        """Try to open all three MMF pages. Returns True on success."""
        try:
            phys_sz  = ctypes.sizeof(SPageFilePhysics)
            graph_sz = ctypes.sizeof(SPageFileGraphic)
            stat_sz  = ctypes.sizeof(SPageFileStatic)

            self._mm_physics  = mmap.mmap(0, phys_sz,  self._MMAP_PHYSICS)
            self._mm_graphics = mmap.mmap(0, graph_sz, self._MMAP_GRAPHICS)
            self._mm_static   = mmap.mmap(0, stat_sz,  self._MMAP_STATIC)

            self._physics  = SPageFilePhysics.from_buffer(self._mm_physics)
            self._graphics = SPageFileGraphic.from_buffer(self._mm_graphics)
            self._static   = SPageFileStatic.from_buffer(self._mm_static)

            self._connected = True
            print("[AC SM] Connected to Assetto Corsa shared memory.")
            return True

        except Exception as exc:
            self._connected = False
            print(f"[AC SM] Could not open shared memory ({exc}). "
                  "Is AC running and in a session?")
            return False

    def _close(self):
        for mm in (self._mm_physics, self._mm_graphics, self._mm_static):
            if mm is not None:
                try:
                    mm.close()
                except Exception:
                    pass
        self._mm_physics  = None
        self._mm_graphics = None
        self._mm_static   = None
        self._physics     = None
        self._graphics    = None
        self._static      = None
        self._connected   = False

    def _poll_loop(self):
        while self._running:
            if not self._connected:
                time.sleep(2.0)
                self._open()
                continue

            try:
                snapshot = self._snapshot()
                with self._lock:
                    self._data = snapshot
            except Exception as exc:
                print(f"[AC SM] Read error: {exc}")
                self._close()

            time.sleep(self.poll_interval)

    def _snapshot(self) -> dict:
        gr = self._graphics
        phys = self._physics
        stat = self._static

        flag_raw  = int(gr.Flag)
        flag_name = _FLAG_NAMES.get(flag_raw, "none")

        return {
            #Static Data
            "track" : str(stat.Track),
            "num_cars" : int(stat.NumCars), 
            "car_model": str(stat.CarModel),

            #Graphic Data
            "connected":               True,
            # Flag
            "flag":                    flag_name,
            "flag_raw":                flag_raw,
            # Timing / position
            "position":                int(gr.Position),
            "completed_laps":          int(gr.CompletedLaps),
            "current_lap_ms":          int(gr.iCurrentTime),
            "last_lap_ms":             int(gr.iLastTime),
            "best_lap_ms":             int(gr.iBestTime),
            "session_time_left":       float(gr.SessionTimeLeft),
            "normalized_car_position": float(gr.NormalizedCarPosition),
            # Pit
            "is_in_pit":               bool(gr.IsInPit),
            "is_in_pit_lane":          bool(gr.IsInPitLane),
            # Session meta
            "status":                  int(gr.Status),
            "session_type":            int(gr.Session),
            # Gaps — not exposed by AC shared memory
            "gap_ahead_ms":            None,
            "gap_behind_ms":           None,
            "gap_to_leader_ms":        None,
            #Tire Compound
            "tire_compound":           str(gr.TyreCompound),
            # Physics — tyre wear (0.0 = new, higher = more worn)
            # Order: FL, FR, RL, RR
            "tyre_wear": [
                round(float(phys.TyreWear[0]), 3),
                round(float(phys.TyreWear[1]), 3),
                round(float(phys.TyreWear[2]), 3),
                round(float(phys.TyreWear[3]), 3),
            ],
            # Physics — tyre dirty level (0.0 = clean, higher = more dirt) FL/FR/RL/RR
            "tyre_dirty": [
                round(float(phys.TyreDirtyLevel[0]), 3),
                round(float(phys.TyreDirtyLevel[1]), 3),
                round(float(phys.TyreDirtyLevel[2]), 3),
                round(float(phys.TyreDirtyLevel[3]), 3),
            ],
 
            # Physics — tyre core temperature in °C FL/FR/RL/RR
            "tyre_core_temp": [
                round(float(phys.TyreCoreTemperature[0]), 1),
                round(float(phys.TyreCoreTemperature[1]), 1),
                round(float(phys.TyreCoreTemperature[2]), 1),
                round(float(phys.TyreCoreTemperature[3]), 1),
            ],

            # Physics — car damage (0.0 = no damage, higher = more damage)
            # Indices: 0=front, 1=rear, 2=left, 3=right, 4=centre
            "car_damage": [
                round(float(phys.CarDamage[0]), 3),
                round(float(phys.CarDamage[1]), 3),
                round(float(phys.CarDamage[2]), 3),
                round(float(phys.CarDamage[3]), 3),
                round(float(phys.CarDamage[4]), 3),
            ],
            #Physics - Fuel level
            "fuel": float(phys.Fuel)
        }


# ---------------------------------------------------------------------------
# Module-level singleton — mirrors the pattern in your udp_client.py
# ---------------------------------------------------------------------------

_instance: AcSharedMemory | None = None


def start_shared_memory(poll_interval: float = 0.05) -> None:
    """Create (if needed) and start the global shared memory reader."""
    global _instance
    if _instance is None:
        _instance = AcSharedMemory(poll_interval=poll_interval)
    _instance.start()


def stop_shared_memory() -> None:
    """Stop the global shared memory reader."""
    global _instance
    if _instance is not None:
        _instance.stop()


def get_shared_memory_data() -> dict:
    """
    Return the latest shared memory snapshot, or {} if not started.
    Safe to call from any thread.
    """
    if _instance is None:
        return {}
    return _instance.get_data()