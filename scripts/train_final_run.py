"""
scripts/train_final_run.py
--------------------------
Master sequential final training run for all 5 models:
  1. LightGBM (GBDT)
  2. XGBoost (GPU Hist)
  3. FlowMLP (Residual MLP)
  4. FT-Transformer (Feature Tokenizer + Transformer)
  5. FlowFusion (Multi-View Gated Architecture)

Runs each model sequentially across all 5 random seeds, aggregates results,
prints a paper-quality comparison table, and saves final comparison outputs.

Usage:
    python scripts/train_final_run.py
    python scripts/train_final_run.py --smoke-test
"""

import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.utils.logger import get_logger
from src.utils.gpu import log_hardware_summary

logger = get_logger("train_final_run")
RESULTS_DIR = ROOT / "experiments" / "results"


def run_script(script_name: str, smoke_test: bool = False) -> bool:
    cmd = [sys.executable, str(ROOT / "scripts" / script_name)]
    if smoke_test:
        cmd.append("--smoke-test")

    logger.info(f"\n{'='*70}")
    logger.info(f"LAUNCHING EXPERIMENT: {script_name} {'(SMOKE TEST)' if smoke_test else ''}")
    logger.info(f"{'='*70}\n")

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    if result.returncode == 0:
        logger.info(f"[OK] {script_name} finished successfully in {elapsed/60:.2f} min.")
        return True
    else:
        logger.error(f"[FAIL] {script_name} failed with return code {result.returncode}.")
        return False


def collect_final_results() -> pd.DataFrame:
    models_info = [
        ("LightGBM",       RESULTS_DIR / "lightgbm" / "results.json"),
        ("XGBoost",        RESULTS_DIR / "xgboost" / "results.json"),
        ("FlowMLP",        RESULTS_DIR / "mlp" / "results.json"),
        ("FT-Transformer", RESULTS_DIR / "ft_transformer" / "results.json"),
        ("FlowFusion",     RESULTS_DIR / "flow_fusion" / "all_gated" / "results.json"),
    ]

    rows = []
    comparison_dict = {}

    for name, path in models_info:
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                test_agg = data.get("test_aggregated", {})
                comparison_dict[name] = data

                row = {
                    "Model":       name,
                    "Accuracy":    f"{test_agg.get('accuracy', {}).get('mean', 0.0):.4f} +/- {test_agg.get('accuracy', {}).get('std', 0.0):.4f}",
                    "Macro-F1":    f"{test_agg.get('macro_f1', {}).get('mean', 0.0):.4f} +/- {test_agg.get('macro_f1', {}).get('std', 0.0):.4f}",
                    "Weighted-F1": f"{test_agg.get('weighted_f1', {}).get('mean', 0.0):.4f} +/- {test_agg.get('weighted_f1', {}).get('std', 0.0):.4f}",
                    "ROC-AUC":     f"{test_agg.get('roc_auc', {}).get('mean', 0.0):.4f} +/- {test_agg.get('roc_auc', {}).get('std', 0.0):.4f}",
                }
                rows.append(row)
            except Exception as e:
                logger.warning(f"Could not parse {path}: {e}")

    df = pd.DataFrame(rows)
    
    csv_path = RESULTS_DIR / "final_comparison.csv"
    df.to_csv(csv_path, index=False)
    
    json_path = RESULTS_DIR / "final_comparison.json"
    with open(json_path, "w") as f:
        json.dump(comparison_dict, f, indent=2)

    return df


def main():
    parser = argparse.ArgumentParser(description="Master final run for DDoS classification benchmark.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-seed 2-epoch smoke test for all models")
    args = parser.parse_args()

    log_hardware_summary()
    t_start = time.time()

    scripts = [
        "train_lightgbm.py",
        "train_xgboost.py",
        "train_mlp.py",
        "train_ft_transformer.py",
        "train_flow_fusion.py",
    ]

    print("\n" + "=" * 70)
    print("      DDoS DETECTION BENCHMARK - FINAL TRAINING RUN      ")
    print("=" * 70 + "\n")

    for script in scripts:
        success = run_script(script, smoke_test=args.smoke_test)
        if not success:
            logger.error(f"Stopping final run pipeline due to failure in {script}")
            sys.exit(1)

    df_results = collect_final_results()

    total_time = (time.time() - t_start) / 60.0

    print("\n" + "=" * 70)
    print(f"  DDoS DETECTION - FINAL RESULTS ({'SMOKE TEST' if args.smoke_test else '5 SEEDS AGGREGATED'})")
    print("=" * 70)
    print("\n" + df_results.to_string(index=False))
    print(f"\nTotal benchmark execution time: {total_time:.2f} minutes")
    print(f"Final results saved to:")
    print(f"  - {RESULTS_DIR / 'final_comparison.csv'}")
    print(f"  - {RESULTS_DIR / 'final_comparison.json'}\n")


if __name__ == "__main__":
    main()
