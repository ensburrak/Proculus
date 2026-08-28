import os
from pathlib import Path

root = Path(r"c:\Users\eness\Downloads\AutoTraderBot\AutoTraderBot")
target_files = [
    Path("risk/daily_risk_budget.py"),
    Path("signal_labeler.py"),
    Path("signal_logger.py"),
    Path("state_manager.py")
]

for rel_path in target_files:
    p = root / rel_path
    if not p.exists():
        continue
    print(f"\n=== Context in {rel_path} ===")
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(lines):
        if "file_lock" in line:
            start = max(0, i - 2)
            end = min(len(lines), i + 4)
            for j in range(start, end):
                print(f"{j+1}: {lines[j]}")
