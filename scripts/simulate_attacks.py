"""
scripts/simulate_attacks.py
----------------------------
Real-Time DDoS Attack Traffic Simulator & API Evaluator.

Simulates network flow traffic by sampling real flow vectors from the test set for
all 6 attack classes (BENIGN, DrDoS_LDAP, DrDoS_MSSQL, DrDoS_NetBIOS, DrDoS_UDP, Syn)
and streaming them sequentially or randomly to the FastAPI deployment endpoint (/predict).

Usage:
    # 1. Start the API server in one terminal:
    #    python -m uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000
    #
    # 2. Run the simulator in another terminal:
    #    python scripts/simulate_attacks.py
    #    python scripts/simulate_attacks.py --num-per-class 5 --delay 0.2
"""

import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PROCESSED_DIR = ROOT / "data" / "processed" / "test"
API_URL = "http://127.0.0.1:8000/predict"


def load_test_samples_per_class(num_per_class: int = 5) -> tuple[dict, list]:
    """Load test flow feature vectors grouped by ground-truth attack class."""
    X = np.load(PROCESSED_DIR / "X.npy", mmap_mode="r")
    y = np.load(PROCESSED_DIR / "y.npy", mmap_mode="r")
    label_classes = list(np.load(PROCESSED_DIR / "label_classes.npy", allow_pickle=True))

    samples_by_class = {}
    rng = np.random.default_rng(42)

    for idx, cname in enumerate(label_classes):
        class_indices = np.where(y == idx)[0]
        selected = rng.choice(class_indices, size=min(num_per_class, len(class_indices)), replace=False)
        samples_by_class[cname] = [X[i].tolist() for i in selected]

    return samples_by_class, label_classes


def run_simulation(num_per_class: int = 5, delay_sec: float = 0.3):
    """Run real-time attack simulation stream."""
    print("=" * 80)
    print("🚀 REAL-TIME DDOS ATTACK SIMULATION STREAM")
    print(f"Target API Endpoint: {API_URL}")
    print("=" * 80)

    # Check if API server is reachable
    try:
        r = requests.get("http://127.0.0.1:8000/health", timeout=3)
        info = r.json()
        print(f"✅ Connected to API | Service: {info.get('service')} | Model: {info.get('model')}\n")
    except Exception as e:
        print(f"❌ Error connecting to API server at http://127.0.0.1:8000/ - {e}")
        print("   Make sure the server is running: python -m uvicorn src.deployment.app:app --port 8000")
        return

    samples_by_class, label_classes = load_test_samples_per_class(num_per_class)

    correct_count = 0
    total_count = 0

    print(f"{'SEC':<6} | {'SIMULATED ATTACK':<16} | {'PREDICTED CLASS':<16} | {'CONFIDENCE':<10} | {'RESULT':<6} | {'LATENCY':<8}")
    print("-" * 80)

    t_start = time.time()

    for cname in label_classes:
        flows = samples_by_class[cname]
        for flow in flows:
            t_curr = time.time() - t_start
            
            try:
                res = requests.post(API_URL, json={"features": flow}, timeout=10)
                if res.status_code == 200:
                    data = res.json()
                    pred_class = data["predicted_class"]
                    conf = data["confidence"] * 100.0
                    lat = data["latency_ms"]

                    is_correct = (pred_class == cname)
                    status = "✅ PASS" if is_correct else "❌ FAIL"
                    if is_correct:
                        correct_count += 1
                    total_count += 1

                    print(f"{t_curr:05.1f}s | {cname:<16} | {pred_class:<16} | {conf:>8.2f}% | {status:<6} | {lat:>6.1f}ms")
                else:
                    print(f"API Error {res.status_code}: {res.text}")
            except Exception as ex:
                print(f"Failed to query API: {ex}")

            time.sleep(delay_sec)

    print("=" * 80)
    acc = (correct_count / total_count * 100.0) if total_count > 0 else 0
    print(f"🎉 SIMULATION COMPLETE | Accuracy: {acc:.2f}% ({correct_count}/{total_count} flows correctly classified)")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate real-time DDoS flow attack stream.")
    parser.add_argument("--num-per-class", type=int, default=5, help="Number of flow samples per class (default 5)")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between flow injections in seconds (default 0.2)")
    args = parser.parse_args()

    run_simulation(num_per_class=args.num_per_class, delay_sec=args.delay)
