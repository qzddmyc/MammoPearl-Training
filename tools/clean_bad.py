# 作用：将 /src/data/bounding-box/*.csv 中的脏数据读取，
# 再对 /data/raw/vindr_detection_folds.csv 中的数据清洗并生成副本。

# 暂不考虑使用此脚本过滤生成的 csv 标注文件，而是在模型初始化时对数据进行一次过滤。

"""Remove bad samples listed in bad_data_record.csv from the raw CSV.

Creates a backup of the original `data/raw/vindr_detection_folds.csv` (banned now) and
writes the cleaned CSV to `data/raw/vindr_detection_folds.cleaned.csv`.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import csv
import pandas as pd
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_bad_record(root: Path) -> Path | None:
    candidates = [
        root / "src" / "data" / "bounding-box" / "bad_data_record_mobilenet.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    root = repo_root()
    bad_path = find_bad_record(root)
    if bad_path is None:
        print("bad_data_record.csv not found; aborting.")
        sys.exit(1)

    raw_csv = root / "data" / "raw" / "vindr_detection_folds.csv"
    if not raw_csv.exists():
        print(f"Raw CSV not found: {raw_csv}")
        sys.exit(1)

    df = pd.read_csv(raw_csv, low_memory=False)

    # read bad pairs
    bad_pairs = set()
    with bad_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id")
            iid = row.get("image_id")
            if pid and iid:
                bad_pairs.add((str(pid), str(iid)))

    if not bad_pairs:
        print("No bad entries found in record; nothing to do.")
        return

    # create mask of rows to keep
    mask = [
        (str(r["patient_id"]), str(r["image_id"])) not in bad_pairs
        for _, r in df.iterrows()
    ]

    removed_count = len(df) - sum(mask)

    # backup original
    backup = raw_csv.with_suffix(raw_csv.suffix + ".bak")
    if not backup.exists():
        # shutil.copy2(raw_csv, backup)
        print(f"Backed up original to: {backup}。注意，并没有新增备份文件，代码被注释了。")
    else:
        print(f"Backup already exists: {backup}")

    cleaned = root / "data" / "raw" / "vindr_detection_folds.cleaned.csv"
    df_clean = df[mask]
    df_clean.to_csv(cleaned, index=False)

    print(f"Removed {removed_count} rows. Cleaned CSV written to: {cleaned}")


if __name__ == "__main__":
    main()
