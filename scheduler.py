import os
import json
import time
import subprocess
import sys
import requests
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil


#Walkthrough of different situations:

#LSTM (GPU Over)


KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY = os.environ["KAGGLE_KEY"]

class workbook_running(Exception):
    pass

def trigger_notebook(notebook_id, enable_gpu,enable_tpu):
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
        print(f"  Pull stderr: {pull.stderr.strip()}")  

        print(f'pull return code is {pull.returncode}')
        print('return code not complete')
        exit_line()

        
    # Step 2 — open the metadata file
    meta_path = f"/tmp/kernel_push/{safe_id}/kernel-metadata.json"
    
    with open(meta_path) as f:
        meta = json.load(f)
        
    meta.pop('docker_image', None)
    meta.pop('machine_shape', None)
    
    if enable_gpu:          
        meta["enable_gpu"] = True
        meta["enable_internet"] = True
        meta["enable_tpu"] = False

    if not enable_gpu:
        meta["enable_gpu"] = False
        meta["enable_internet"] = True
        meta['machine_shape'] = None
        meta['keywords'] = []

        if enable_tpu:
            meta["enable_tpu"] = True
        else:
            meta["enable_tpu"] = False
            
        
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        print(meta)

    
    if enable_gpu:    
        push = subprocess.run(
            ["kaggle", "kernels", "push", "-p", f"/tmp/kernel_push/{safe_id}", "--accelerator", "NvidiaTeslaP100"],
            capture_output=True, text=True
        )

    elif enable_tpu:

        push = subprocess.run(
            ["kaggle", "kernels", "push", "-p", f"/tmp/kernel_push/{safe_id}", "--accelerator", "tpuVmV5e8"],
            capture_output=True, text=True
        )
        
    else:
        push = subprocess.run(
            ["kaggle", "kernels", "push", "-p", f"/tmp/kernel_push/{safe_id}"],
            capture_output=True, text=True
        )

    combined = (push.stdout + push.stderr).lower()
    
    print(f'combined value:{combined}')
    
    if any(word in combined for word in ["quota", "exceeded", "limit reached", "no gpu","no tpu"]):
        print('quota exceeded')
        return "quota_exceeded"
        
    if push.returncode != 0:
        print(f'push return code is {push.returncode}')
        print('return code not complete')
        exit_line()

        
    return "ok"
    
def get_notebook_status(notebook_id):
    result = subprocess.run(
        ["kaggle", "kernels", "status", notebook_id],
        capture_output=True, text=True
    )
    status = result.stdout.lower()

    status_map = {
        "running": "running",
        "complete": "complete",
        "error": "error",
        "cancel_acknowledged": "cancelacknowledged",
        "queued": "queued"
    }

    for key, value in status_map.items():
        if key in status:
            print(f'this is the status:{key}')
            return key
            


def watch_notebook(notebook_id, allow_gpu,label):
    gpu_gone = False
    tpu_gone = False
    
    def trigger():
        #Check if GPU is allowed (False for XGB and True for LSTM) and whether quota is exceeded
        nonlocal gpu_gone, tpu_gone  # tpu_gone needs to be nonlocal too

        if allow_gpu:
            # Try GPU first if not already exhausted
            if not gpu_gone:
                result = trigger_notebook(notebook_id, enable_gpu=True, enable_tpu=False)
                if result == "quota_exceeded":
                    print('GPU quota exceeded')
                    gpu_gone = True
                else:
                    return  # GPU worked fine, done

            # GPU is gone — try TPU on Thursdays
            if (datetime.today().weekday() == 3 or datetime.today().weekday() == 4) and not tpu_gone:
                result = trigger_notebook(notebook_id, enable_gpu=False, enable_tpu=True)
                if result == "quota_exceeded":
                    print('TPU quota exceeded')
                    tpu_gone = True
                else:
                    return  # TPU worked fine, done

            # Both GPU and TPU gone (or not Thursday/Friday) — fall back to CPU
            trigger_notebook(notebook_id, enable_gpu=False, enable_tpu=False)

        else:
            # XGB — CPU only, always
            trigger_notebook(notebook_id, enable_gpu=False, enable_tpu=False)

    #Check Status of notebook to - if unknown, begin run, otherwise carry on
    
    status = get_notebook_status(notebook_id)
    
    if status == 'running':
        raise workbook_running('Workbook still running')

    trigger()

    run_start = datetime.now(timezone.utc)

    while True:
        
        time.sleep(60)
        
        status = get_notebook_status(notebook_id)

        print(f'running status: {status}')

        elapsed = (datetime.now(timezone.utc) - run_start).total_seconds() / 3600
        mode = "GPU" if (allow_gpu and not gpu_gone) else "CPU"
        print(f"[{label}] Status: {status} | Elapsed: {elapsed:.2f}h | Mode: {mode}")
        
        if status != "running" and status!= 'queued':
            print(f'status error: status is {status}')
            break


def is_workflow_already_running(workflow_file):
    github_token = os.environ.get("GH_PAT")
    github_repo = os.environ.get("GITHUB_REPOSITORY")
    
    result = subprocess.run([
        "curl", "-s",
        "-H", f"Authorization: token {github_token}",
        "-H", "Accept: application/vnd.github.v3+json",
        f"https://api.github.com/repos/{github_repo}/actions/workflows/{workflow_file}/runs?status=in_progress"
    ], capture_output=True, text=True)
    
    data = json.loads(result.stdout)
    print(data)
    runs = data.get("workflow_runs", [])

    print(len(runs))
    return len(runs) > 1

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--gpu", action="store_true", help="Allow GPU usage")
    parser.add_argument("--label", required=True, help="Display label e.g. LSTM Trainer")
    parser.add_argument("--workflow", required=True) 
    args = parser.parse_args()

    if is_workflow_already_running(args.workflow):
        print("Another instance already running — exiting", flush=True)
        sys.exit(0)
        
    else:
        try:
            watch_notebook(f"{KAGGLE_USERNAME}/{args.notebook}", allow_gpu=args.gpu, label=args.label)

        except workbook_running:
            print('Workbook running: exiting gracefully')
            sys.exit(0)
            
        except Exception as e:
            print(e)
            sys.exit(1)
            time.sleep(30)

