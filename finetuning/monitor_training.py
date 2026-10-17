"""
Live Training Status Monitor for BGE-Reranker-Base
===================================================
Run this anytime in your terminal to see live progress, loss, and validation scores:
    .\.venv_gpu\Scripts\python.exe monitor_training.py
"""

import sys
import re
from pathlib import Path

LOG_PATH = Path(r"C:\Users\Shilpa\.gemini\antigravity-ide\brain\461c955e-1444-44f8-9461-d8db1d04b463\.system_generated\tasks\task-620.log")

def get_latest_stats():
    if not LOG_PATH.exists():
        print(f"Log file not found at: {LOG_PATH}")
        return

    text = LOG_PATH.read_text(encoding="utf-8", errors="ignore")
    
    # Check baseline
    baseline_match = re.search(r"Pretrained Baseline -> MRR@10:\s+([\d\.]+)%\s+\|\s+MAP:\s+([\d\.]+)%", text)
    baseline_mrr = baseline_match.group(1) if baseline_match else "N/A"
    baseline_map = baseline_match.group(2) if baseline_match else "N/A"

    # Find all validation steps
    val_evals = re.findall(r"\[Step\s+([\d,]+)/([\d,]+)\] Running Validation Evaluation[\s\S]*?Validation -> MRR@10:\s+([\d\.]+)%\s+\|\s+MAP:\s+([\d\.]+)%", text)
    
    # Find latest progress line
    # Format: Epoch 1/1:  13%|#3        | 2514/18745 [1:16:44<9:12:53,  2.04s/batch, loss=0.3323, lr=1.82e-05]
    progress_matches = re.findall(r"Epoch\s+1/1:\s+(\d+)%\|.*?\|\s+(\d+)/(\d+)\s+\[([^<]+)<([^,]+),\s+([\d\.]+)s/batch(?:,\s+loss=([\d\.]+))?(?:,\s+lr=([e\d\.-]+))?\]", text)

    print("=" * 65)
    print("       BGE-RERANKER-BASE TRAINING MONITOR")
    print("=" * 65)
    print(f"  Pretrained Baseline : MRR@10: {baseline_mrr}% | MAP: {baseline_map}%")
    
    if val_evals:
        print("\n  [Validation Milestones]:")
        for step, total, mrr, v_map in val_evals:
            print(f"    - Step {step}/{total}: MRR@10 = {mrr}% | MAP = {v_map}% (Checkpoint Saved)")
    else:
        print("\n  [Validation Milestones]: None yet (first at Step 2,500)")

    if progress_matches:
        pct, curr_step, total_step, elapsed, eta, speed, loss, lr = progress_matches[-1]
        print("\n  [Current Progress]:")
        print(f"    Step:           {curr_step} / {total_step} ({pct}%)")
        print(f"    Elapsed Time:   {elapsed.strip()}")
        print(f"    Estimated ETA:  {eta.strip()}")
        print(f"    Speed:          {speed}s / batch")
        if loss:
            print(f"    Current Loss:   {loss}")
        if lr:
            print(f"    Learning Rate:  {lr}")
    
    print("=" * 65)
    print(f"Log location: {LOG_PATH}")

if __name__ == "__main__":
    get_latest_stats()
