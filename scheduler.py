import os
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

KAGGLE_USERNAME = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY = os.environ["KAGGLE_KEY"]

NOTEBOOK_A_ID = f"{KAGGLE_USERNAME}/lstm-trainer-condensed"
NOTEBOOK_B_ID = f"{KAGGLE_USERNAME}/xgb-trainer"


def trigger_notebook(notebook_id: str, enable_gpu: bool) -> str:
    print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Triggering {notebook_id} | GPU={enable_gpu}")

    pull = subprocess.run(
        ["kaggle", "kernels", "pull", notebook_id, "-p", "/tmp/kernel_push", "-m"],
        capture_output=True, text=True
    )

    # if pull.returncode != 0:
    #     print(f"  ❌ Pull failed: {pull.stderr.strip()}")
    #     return "error"

    # # Step 2 — open the metadata file
    # meta_path = "/tmp/kernel_push/kernel-metadata.json"
    # with open(meta_path) as f:
    #     meta = json.load(f)
    # meta["enable_gpu"] = enable_gpu
    # with open(meta_path, "w") as f:
    #     json.dump(meta, f, indent=2)

    # push = subprocess.run(
    #     ["kaggle", "kernels", "push", "-p", "/tmp/kernel_push"],
    #     capture_output=True, text=True
    # )



    result = subprocess.run(
        ["kaggle", "kernels", "status", f"{KAGGLE_USERNAME}/lstm-trainer-condensed"],
        capture_output=True,
        text=True
    )

    print(result.stdout)




trigger_notebook(NOTEBOOK_A_ID, enable_gpu = True)





