import os
import json
import time
import subprocess
import threading
import sys
from datetime import datetime, timezone
from pathlib import Path
import shutil

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY = os.environ["KAGGLE_KEY"]
NOTEBOOK_A_ID = f"{KAGGLE_USERNAME}/lstm-trainer-condensed"
NOTEBOOK_B_ID = f"{KAGGLE_USERNAME}/xgb-trainer"

fatal_error = threading.Event()


def trigger_notebook(notebook_id, enable_gpu):
    safe_id = notebook_id.replace("/", "_")

    if Path(f"/tmp/kernel_push/{safe_id}").exists():
        shutil.rmtree(f"/tmp/kernel_push/{safe_id}")
        
    Path(f"/tmp/kernel_push/{safe_id}").mkdir(parents=True)
    
    print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Triggering {notebook_id} | GPU={enable_gpu}")

    pull = subprocess.run(
        ["kaggle", "kernels", "pull", notebook_id, "-p", f"/tmp/kernel_push/{safe_id}", "-m"],
        capture_output=True, text=True
    )

    
    if pull.returncode != 0:
        print(f"  Pull returncode: {pull.returncode}")
        print(f"  Pull stdout: {pull.stdout.strip()}")
        print(f"  Pull stderr: {pull.stderr.strip()}")  # ← this will tell you exactly why

        print(f'pull return code is {pull.returncode}')
        print('return code not complete')
        fatal_error.set()
        exit_line()

        
    # Step 2 — open the metadata file
    meta_path = f"/tmp/kernel_push/{safe_id}/kernel-metadata.json"
    
    with open(meta_path) as f:
        meta = json.load(f)
        
    meta["enable_gpu"] = enable_gpu
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    push = subprocess.run(
        ["kaggle", "kernels", "push", "-p", f"/tmp/kernel_push/{safe_id}"],
        capture_output=True, text=True
    )

    combined = (push.stdout + push.stderr).lower()
    print(combined)

    if any(word in combined for word in ["quota", "exceeded", "limit reached", "no gpu"]):
        return "quota_exceeded"
        
    if push.returncode != 0:
        print(f'push return code is {push.returncode}')
        print('return code not complete')
        fatal_error.set()
        
    return "ok"
    
def get_notebook_status(notebook_id):
    result = subprocess.run(
        ["kaggle", "kernels", "status", notebook_id],
        capture_output=True, text=True
    )
    status = result.stdout.lower()
    print(f'this is the status:{status}')
    return status


def watch_notebook(notebook_id, allow_gpu,label):
    gpu_gone = False
    def trigger():
        #Check if GPU is allowed (False for XGB and True for LSTM) and whether quota is exceeded
        nonlocal gpu_gone 
        
        if allow_gpu and not gpu_gone:
            result = trigger_notebook(notebook_id, enable_gpu=True)
            if result == "quota_exceeded":
                gpu_gone = True
                return trigger_notebook(notebook_id, enable_gpu=False)
            else:
                return 
        else:
            #Exclusively for XGB
            return trigger_notebook(notebook_id, enable_gpu=False)
            
    #Check Status of notebook to - if unknown, begin run, otherwise carry on
    # status = get_notebook_status(notebook_id)
    # if status != "unknown":
    #     fatal_error.set()
    #     exit_line()

    trigger()
    
    run_start = datetime.now(timezone.utc)

    while True:
        
        time.sleep(60)
        
        status = get_notebook_status(notebook_id)

        print(f'running status: {status}')

        elapsed = (datetime.now(timezone.utc) - run_start).total_seconds() / 3600
        mode = "GPU" if (allow_gpu and not gpu_gone) else "CPU"
        print(f"[{label}] Status: {status} | Elapsed: {elapsed:.2f}h | Mode: {mode}")
        
        if status != "running":
            print(f'status error: status is {status}')
            fatal_error.set()
            break

        time.sleep(60)


if __name__ == "__main__":
    thread_a = threading.Thread(
        target=watch_notebook,
        args=(NOTEBOOK_A_ID, True, "LSTM Trainer"),   # allow_gpu=True
        daemon=False
    )
    
    thread_b = threading.Thread(
        target=watch_notebook,
        args=(NOTEBOOK_B_ID, False, "XGB Trainer"),   # allow_gpu=False
        daemon=False
    )


    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    if fatal_error.is_set():

        print("\n❌ Fatal error encountered — exiting with error code")
        sys.exit(1)




