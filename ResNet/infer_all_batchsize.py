import subprocess

batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]

for batch in batch_sizes:

    cmd = [
        "python",
        "infer_5060.py",
        "--batch",
        str(batch)
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    # 正常輸出
    with open(f"batch_{batch}.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout)

    # 有錯誤
    if result.returncode != 0:

        with open(f"batch_{batch}_error.txt", "w", encoding="utf-8") as f:
            f.write(result.stderr)

        print(f"Batch {batch} failed")

        break

    print(f"Saved batch {batch}")