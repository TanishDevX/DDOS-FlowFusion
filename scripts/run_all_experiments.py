"""
scripts/run_all_experiments.py
-------------------------------
Master orchestration script — runs the full experiment pipeline end-to-end.

Pipeline order:
  Step 1 : Preprocess raw CSVs   → data/processed/
  Step 2 : Train XGBoost         → models/xgb_best.joblib
  Step 3 : Train LightGBM        → models/lgbm_best.joblib
  Step 4 : Train MLP             → models/mlp_best.keras
  Step 5 : Train FT-Transformer  → models/ft_transformer_best.keras
  Step 6 : Evaluate all models   → experiments/results/comparison/

Each step is timed and any failure is logged without stopping remaining steps.
A final summary table is printed and saved to experiments/results/pipeline_summary.json.

Run from project root:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --skip-preprocess   # if data already processed
    python scripts/run_all_experiments.py --models xgboost lightgbm
"""

import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger

logger = get_logger("run_all_experiments")


# ═══════════════════════════════════════════════════════════════════════════
# Step runner
# ═══════════════════════════════════════════════════════════════════════════

def run_step(step_name: str, script_path: Path, extra_args: list = None) -> dict:
    """
    Run a Python script as a subprocess and return timing + status.

    Returns:
        {"step": str, "status": "ok"|"failed", "duration_s": float, "error": str|None}
    """
    extra_args = extra_args or []
    cmd = [sys.executable, str(script_path)] + extra_args

    logger.info("\n" + "═" * 70)
    logger.info(f"STEP: {step_name}")
    logger.info(f"CMD : {' '.join(cmd)}")
    logger.info("═" * 70)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            check=True,          # raises CalledProcessError on non-zero exit
            capture_output=False # stream stdout/stderr directly to console
        )
        duration = time.time() - t0
        logger.info(f"\n✅  {step_name} completed in {duration/60:.1f} min")
        return {"step": step_name, "status": "ok", "duration_s": duration, "error": None}
    except subprocess.CalledProcessError as e:
        duration = time.time() - t0
        err_msg  = str(e)
        logger.error(f"\n❌  {step_name} FAILED after {duration/60:.1f} min: {err_msg}")
        return {"step": step_name, "status": "failed", "duration_s": duration, "error": err_msg}
    except Exception as e:
        duration = time.time() - t0
        logger.error(f"\n❌  {step_name} FAILED: {e}")
        return {"step": step_name, "status": "failed", "duration_s": duration, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run the full DDoS experiment pipeline.")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip preprocessing (use if data/processed/ already exists).",
    )
    parser.add_argument(
        "--models", nargs="+",
        choices=["xgboost", "lightgbm", "mlp", "ft_transformer", "flow_fusion"],
        default=["xgboost", "lightgbm", "mlp", "ft_transformer", "flow_fusion"],
        help="Which models to train (default: all five).",
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip the final evaluation step.",
    )
    args = parser.parse_args()

    pipeline_start = time.time()
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\nPipeline started at {run_ts}")
    logger.info(f"Models to train: {args.models}")
    logger.info(f"Skip preprocess: {args.skip_preprocess}")

    step_results = []
    scripts_dir  = ROOT / "scripts"

    # ── Step 1: Preprocess ─────────────────────────────────────────────────
    if not args.skip_preprocess:
        result = run_step(
            "Preprocessing (data/raw → data/processed)",
            scripts_dir / "preprocess.py",
        )
        step_results.append(result)
        if result["status"] == "failed":
            logger.error("Preprocessing failed — cannot train without processed data. Aborting.")
            _save_and_print_summary(step_results, pipeline_start, run_ts)
            sys.exit(1)
    else:
        logger.info("\n[SKIP] Preprocessing (--skip-preprocess flag set)")
        step_results.append({
            "step": "Preprocessing", "status": "skipped",
            "duration_s": 0, "error": None,
        })

    # ── Step 2-5: Train models ─────────────────────────────────────────────
    model_scripts = {
        "xgboost":        scripts_dir / "train_xgboost.py",
        "lightgbm":       scripts_dir / "train_lightgbm.py",
        "mlp":            scripts_dir / "train_mlp.py",
        "ft_transformer": scripts_dir / "train_ft_transformer.py",
        "flow_fusion":    scripts_dir / "train_flow_fusion.py",
    }

    for model_name in args.models:
        script = model_scripts[model_name]
        result = run_step(
            f"Training {model_name.upper()}",
            script,
        )
        step_results.append(result)
        # Training failures are non-fatal — continue with remaining models

    # ── Step 6: Evaluate ───────────────────────────────────────────────────
    if not args.skip_eval:
        # Only evaluate models that were requested AND trained successfully
        trained_models = [
            r["step"].replace("Training ", "").lower()
            for r in step_results
            if r["status"] == "ok" and r["step"].startswith("Training")
        ]
        # Normalize names back
        name_map = {
            "xgboost": "xgboost",
            "lightgbm": "lightgbm",
            "mlp": "mlp",
            "ft-transformer": "ft_transformer",
        }
        trained_models = [name_map.get(n, n) for n in trained_models]

        if trained_models:
            result = run_step(
                "Evaluating all models",
                scripts_dir / "evaluate.py",
                extra_args=["--model", "all"],
            )
            step_results.append(result)
        else:
            logger.warning("No models trained successfully — skipping evaluation.")
    else:
        logger.info("\n[SKIP] Evaluation (--skip-eval flag set)")

    _save_and_print_summary(step_results, pipeline_start, run_ts)


def _save_and_print_summary(step_results: list, pipeline_start: float, run_ts: str):
    total_time = time.time() - pipeline_start

    logger.info("\n" + "═" * 70)
    logger.info("PIPELINE SUMMARY")
    logger.info("═" * 70)

    for r in step_results:
        status_icon = {"ok": "✅", "failed": "❌", "skipped": "⏭️"}.get(r["status"], "?")
        duration_str = f"{r['duration_s']/60:.1f} min" if r["duration_s"] else "—"
        logger.info(f"  {status_icon}  {r['step']:<45}  {r['status']:<8}  {duration_str}")
        if r["error"]:
            logger.info(f"       Error: {r['error'][:120]}")

    logger.info(f"\n  Total pipeline time: {total_time/60:.1f} min")
    logger.info("═" * 70)

    summary = {
        "run_timestamp":    run_ts,
        "total_duration_s": total_time,
        "steps":            step_results,
    }
    summary_path = ROOT / "experiments" / "results" / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Pipeline summary saved → {summary_path}")


if __name__ == "__main__":
    main()
