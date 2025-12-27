import os
from typing import Optional, List
from datetime import datetime

def resolve_latest_backtest_dir(backtests_root: str) -> Optional[str]:
    """Return the most recently created backtest directory, if any.

    Parameters
    ----------
    backtests_root : str
        Absolute path to the ``backtests`` folder that contains run
        snapshots.

    Returns
    -------
    Optional[str]
        Full path to the newest run folder, or ``None`` when the root is
        missing or contains no runs.
    """
    if not os.path.isdir(backtests_root):
        return None
    candidates: List[str] = []
    for entry in os.listdir(backtests_root):
        full_path = os.path.join(backtests_root, entry)
        if not os.path.isdir(full_path):
            continue
        if entry.endswith("_patterns") or entry == "patterns":
            continue
        candidates.append(full_path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: os.path.basename(path))
    return candidates[-1]

def resolve_log_dir(algo_dir: str, subfolder_name: str) -> str:
    """Resolve the directory where logs should be stored.
    
    It attempts to find the latest backtest directory. If found, it creates
    a subfolder there. If not found, it creates a timestamped folder in
    the backtests root.

    Parameters
    ----------
    algo_dir : str
        Root directory of the algorithm.
    subfolder_name : str
        Name of the subfolder to create (e.g., "patterns" or "price_actions").

    Returns
    -------
    str
        Absolute path to the resolved log directory.
    """
    base_time = datetime.now()
    timestamp = base_time.strftime("%Y-%m-%d_%H-%M-%S")
    backtests_root = os.path.join(algo_dir, "backtests")
    latest_backtest_dir = resolve_latest_backtest_dir(backtests_root)
    
    if latest_backtest_dir is None:
        # Fallback: create a new timestamped folder if none exists
        # Note: We append the subfolder name to the timestamp to avoid collision/confusion
        # or we can just create the timestamp folder and put the subfolder inside.
        # Let's mirror the structure: backtests/<timestamp>/<subfolder>
        log_dir = os.path.join(backtests_root, timestamp, subfolder_name)
    else:
        # Use the most recent run folder
        log_dir = os.path.join(latest_backtest_dir, subfolder_name)
        
    return log_dir
