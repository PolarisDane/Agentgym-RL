import os
import time
import subprocess
import glob

# Configuration
SEARCH_DIR = "."
WANDB_API_KEY = "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"
SYNC_INTERVAL = 300  # 5 minutes
ACTIVE_THRESHOLD = 600  # 10 minutes (Considered active if modified within this time)

def get_all_offline_runs():
    # Find all 'offline-run-*' directories recursively in the search directory
    offline_runs = glob.glob(os.path.join(SEARCH_DIR, "**", "offline-run-*"), recursive=True)
    return offline_runs

def is_run_active(run_dir, threshold=ACTIVE_THRESHOLD):
    # A run is considered active if its .wandb file has been modified recently
    wandb_files = glob.glob(os.path.join(run_dir, "*.wandb"))
    if not wandb_files:
        return False
    
    current_time = time.time()
    for f in wandb_files:
        try:
            # Check modification time of the .wandb binary log file
            if current_time - os.path.getmtime(f) < threshold:
                return True
        except OSError:
            continue
    return False

def sync_wandb():
    all_runs = get_all_offline_runs()
    if not all_runs:
        print(f"No offline wandb runs found in {SEARCH_DIR}.")
        return

    # Filter for active runs (those still being written to)
    active_runs = [run for run in all_runs if is_run_active(run)]
    
    if not active_runs:
        print(f"No active runs found (no .wandb files modified in the last {ACTIVE_THRESHOLD}s).")
        return

    # Sort active runs by modification time (newest first) and take latest 5
    active_runs = sorted(active_runs, key=os.path.getmtime, reverse=True)[:5]
    
    print(f"Found {len(active_runs)} active runs to sync.")
    
    # Run wandb sync for each active run
    env = os.environ.copy()
    env["WANDB_API_KEY"] = WANDB_API_KEY
    
    # Sync from oldest to newest among the active ones
    for offline_run in reversed(active_runs):
        print(f"Syncing: {offline_run}")
        try:
            # Sync individual offline run directory
            subprocess.run(["wandb", "sync", offline_run], env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error syncing {offline_run}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during sync: {e}")

if __name__ == "__main__":
    print(f"Starting wandb sync script (Active Runs Only). Interval: {SYNC_INTERVAL}s")
    while True:
        sync_wandb()
        print(f"Sleeping for {SYNC_INTERVAL}s...")
        time.sleep(SYNC_INTERVAL)
