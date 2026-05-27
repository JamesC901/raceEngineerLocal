import os
from dotenv import load_dotenv
 
load_dotenv()
 
 
def read_car_positions() -> list[dict]:
    """Read per-car positions written by the acgaps AC Python app."""
    _GAPS_FILE = os.getenv("_GAPS_FILEPATH", "None")
    try:
        with open(_GAPS_FILE, 'r') as f:
            lines = f.read().strip().splitlines()
    except FileNotFoundError:
        print("\n\nFile Not Found!\n\n")
        return []
 
    cars = []
    for line in lines:
        parts = line.split(',')
        if len(parts) < 7:          # need all 7 columns
            continue
        cars.append({
            'car_index':   int(parts[0]),
            'spline_pos':  float(parts[1]),
            'lap_count':   int(parts[2]),
            'lap_time_ms': int(parts[3]),
            'last_lap_ms': int(parts[4]),
            'best_lap_ms': int(parts[5]),   # column 5 — was wrongly used as is_player
            'is_player':   bool(int(parts[6])),  # column 6 — the actual flag
        })
    return cars
 
 
def calculate_gaps() -> dict:
    """
    Returns gaps in seconds rounded to 2 decimal places, e.g. 0.06.
    Returns None for a gap if not calculable.
    """
    cars = read_car_positions()
    if not cars:
        return {"gap_ahead_s": None, "gap_behind_s": None, "gap_to_leader_s": None}
 
    def race_distance(car):
        return car["lap_count"] + car["spline_pos"]
 
    sorted_cars = sorted(cars, key=race_distance, reverse=True)
 
    player = next((c for c in sorted_cars if c["is_player"]), None)
    if player is None:
        return {"gap_ahead_s": None, "gap_behind_s": None, "gap_to_leader_s": None}
 
    player_idx = sorted_cars.index(player)
 
    def delta_s(car_a, car_b) -> float:
        """
        Gap between two cars in seconds, using the average of their lap times
        as the reference lap duration. Falls back to 90s if no lap data yet.
        """
        lap_a = car_a["last_lap_ms"] or car_a["best_lap_ms"] or 0
        lap_b = car_b["last_lap_ms"] or car_b["best_lap_ms"] or 0
 
        if lap_a and lap_b:
            ref_lap_ms = (lap_a + lap_b) / 2
        elif lap_a:
            ref_lap_ms = lap_a
        elif lap_b:
            ref_lap_ms = lap_b
        else:
            ref_lap_ms = 90_000
 
        spline_delta = abs(race_distance(car_a) - race_distance(car_b))
        gap_ms = spline_delta * ref_lap_ms
        return round(gap_ms / 1000, 2)
 
    car_ahead  = sorted_cars[player_idx - 1] if player_idx > 0 else None
    car_behind = sorted_cars[player_idx + 1] if player_idx < len(sorted_cars) - 1 else None
    leader     = sorted_cars[0]
 
    return {
        "gap_ahead_s":     delta_s(player, car_ahead)  if car_ahead  else None,
        "gap_behind_s":    delta_s(player, car_behind) if car_behind else None,
        "gap_to_leader_s": delta_s(player, leader)     if player_idx > 0 else 0.0,
    }