"""
Assetto Corsa Game Server — Shared Memory only.
All telemetry is read from AC's named memory-mapped files via ac_shared_memory.
UDP remote telemetry has been removed.
"""
 
from ac_shared_memory import start_shared_memory, stop_shared_memory, get_shared_memory_data  # noqa: F401
 
_started = False
 
 
def start_game_server():
    """Start the shared memory reader. Returns None (no background thread needed)."""
    global _started
    if not _started:
        start_shared_memory()
        _started = True
        print("[Game Server] Shared memory reader started.")
    return None
 
 
def stop_game_server():
    """Stop the shared memory reader."""
    global _started
    stop_shared_memory()
    _started = False
    print("[Game Server] Shared memory reader stopped.")
 