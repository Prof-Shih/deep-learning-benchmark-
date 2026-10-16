#可以跑不同batchsize的HEF
import time
import subprocess
import sys
from pathlib import Path


# =========================================================
# Settings
# =========================================================
SCRIPT = "infer_hef.py"

HEF_PATH = "mobilenet_v2_1.0.hef"
IMAGE_DIR = "../Test"

LIMIT = 512

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

OUT_DIR = Path("batch_results")
OUT_DIR.mkdir(exist_ok=True)


for batch in batch_sizes:
    print(f"\n========== Running batch {batch} ==========")

    cmd = [
        sys.executable,
        SCRIPT,
        "--hef", HEF_PATH,
        "--image-dir", IMAGE_DIR,
        "--batch", str(batch),
        "--limit", str(LIMIT),
        "--timeout-ms", "30000",
        "--input-format", "uint8",
        "--output-format", "uint8",
    ]

    print("Command:")
    print(" ".join(cmd))

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    out_file = OUT_DIR / f"batch_{batch}.txt"
    err_file = OUT_DIR / f"batch_{batch}_error.txt"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(result.stdout)

    if result.stderr.strip():
        with open(err_file, "w", encoding="utf-8") as f:
            f.write(result.stderr)

    if result.returncode != 0:
        print(f"Batch {batch} failed.")
        print(f"stdout saved to: {out_file}")
        print(f"stderr saved to: {err_file}")
        break

    print(f"Batch {batch} done.")
    print(f"Saved: {out_file}")

    # 很重要：讓 Hailo device / server 有時間釋放
    time.sleep(3)


print("\n========== ALL DONE ==========")
print(f"Results folder: {OUT_DIR}")