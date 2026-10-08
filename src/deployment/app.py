"""
src/deployment/app.py
---------------------
FastAPI REST API for DDoS detection inference.

Endpoints:
  GET  /              → Health check
  GET  /info          → Model info (classes, feature count, version, model type)
  POST /predict       → Predict class from a flow feature vector
  POST /explain       → Predict + SHAP attributions (tree models) or
                        Integrated Gradients (DL models) + gate values (FlowFusion)

FlowFusion gate values:
  When the loaded model is FlowFusion (all_gated config), the /explain response
  includes `gate_values` — 3 scalar weights showing how much the model trusted
  each branch (tabular, spatial, structural) for this specific prediction.
  e.g. {"tabular": 0.71, "spatial": 0.12, "structural": 0.17}

Run from project root:
    uvicorn src.deployment.app:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000/docs
"""

import time
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import joblib
import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent.parent
MODELS_DIR    = ROOT / "models"
CONFIG_PATH   = ROOT / "config" / "config.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"
STATIC_DIR    = Path(__file__).resolve().parent / "static"

# ── Config ─────────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

_VERSION    = _cfg["project"]["version"]
_MODEL_NAME = "XGBoost"   # Primary deployed model — updated at startup
_IS_FLOW_FUSION = False   # Set True when FlowFusion model loaded
_FLOW_FUSION_BRANCH_NAMES = ["tabular", "spatial", "structural"]

# ── Lazy-load models at startup ────────────────────────────────────────────
_model       = None
_preprocessor= None
_label_classes: Optional[np.ndarray] = None
_feature_names: Optional[np.ndarray] = None


def _load_artifacts():
    global _model, _preprocessor, _label_classes, _feature_names
    global _MODEL_NAME, _IS_FLOW_FUSION

    # Check for FlowFusion v2 model first (proposed model), then v1, then XGBoost
    ff_v2_path = MODELS_DIR / "flow_fusion_v2_all_gated_best.keras"
    ff_v1_path = MODELS_DIR / "flow_fusion_all_gated_best.keras"
    xgb_path   = MODELS_DIR / "xgb_best.joblib"

    # Check for FlowFusion v2 model first, then v1, then XGBoost
    ff_v2_path = MODELS_DIR / "flow_fusion_v2_all_gated_best.keras"
    ff_v1_path = MODELS_DIR / "flow_fusion_all_gated_best.keras"
    xgb_path   = MODELS_DIR / "xgb_best.joblib"

    loaded = False
    for target_path, name in [(ff_v2_path, "FlowFusion v2 (all_gated)"), (ff_v1_path, "FlowFusion v1 (all_gated)")]:
        if target_path.exists():
            try:
                import tensorflow as tf
                from src.models.flow_fusion import FeaturePadding, ClassWeightedFocalLoss, SmoothedSparseCCE
                from src.models.ft_transformer import FeatureTokenizer, TransformerBlock, StochasticDepth
                _model = tf.keras.models.load_model(
                    str(target_path),
                    custom_objects={
                        "FeatureTokenizer":       FeatureTokenizer,
                        "TransformerBlock":       TransformerBlock,
                        "StochasticDepth":        StochasticDepth,
                        "FeaturePadding":         FeaturePadding,
                        "ClassWeightedFocalLoss": ClassWeightedFocalLoss,
                        "SmoothedSparseCCE":      SmoothedSparseCCE,
                    },
                    safe_mode=False,
                    compile=False,
                )
                _MODEL_NAME     = name
                _IS_FLOW_FUSION = True
                loaded          = True
                break
            except Exception as e:
                print(f"Warning: Failed to load {name} from {target_path.name}: {e}")

    if not loaded:
        if xgb_path.exists():
            _model      = joblib.load(xgb_path)
            _MODEL_NAME = "XGBoost"
            _IS_FLOW_FUSION = False
        else:
            model_path = MODELS_DIR / _cfg["deployment"]["model_path"].split("/")[-1]
            if model_path.exists():
                _model = joblib.load(model_path)
                _MODEL_NAME = "XGBoost"
                _IS_FLOW_FUSION = False

    scaler_path = MODELS_DIR / "preprocessor.joblib"
    loaded_scaler = False
    if scaler_path.exists():
        try:
            _preprocessor = joblib.load(scaler_path)
            loaded_scaler = True
        except Exception as e:
            print(f"Warning: Could not load preprocessor.joblib ({e}). Re-fitting StandardScaler from train set.")
    
    if not loaded_scaler:
        from sklearn.preprocessing import StandardScaler
        train_x_path = PROCESSED_DIR / "train" / "X.npy"
        if train_x_path.exists():
            X_tr_mmap = np.load(train_x_path, mmap_mode="r")
            idx = np.random.default_rng(42).choice(len(X_tr_mmap), size=min(150000, len(X_tr_mmap)), replace=False)
            _preprocessor = StandardScaler()
            _preprocessor.fit(X_tr_mmap[idx])

    label_path = PROCESSED_DIR / "train" / "label_classes.npy"
    feat_path  = PROCESSED_DIR / "train" / "feature_names.npy"

    if label_path.exists():
        _label_classes = np.load(label_path, allow_pickle=True)
    if feat_path.exists():
        _feature_names = np.load(feat_path, allow_pickle=True)


# ═══════════════════════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="DDoS Detection API",
    description=(
        "Real-time DDoS detection using CICFlowMeter flow features. "
        "Provides `/predict` and `/explain` (SHAP) endpoints."
    ),
    version=_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    _load_artifacts()


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════

class FlowSample(BaseModel):
    """A single network flow, described by CICFlowMeter feature values."""
    features: List[float] = Field(
        ...,
        description="List of numeric feature values in the same order as feature_names.",
        example=[0.0] * 10,
    )


class PredictResponse(BaseModel):
    predicted_class:   str
    predicted_index:   int
    confidence:        float
    all_probabilities: dict
    model_type:        str
    latency_ms:        float


class ExplainResponse(BaseModel):
    predicted_class:   str
    confidence:        float
    model_type:        str
    top_features:      List[dict]          # [{feature, shap_value|attribution}, ...]
    gate_values:       Optional[dict]      # FlowFusion only: {branch: scalar_gate}
    latency_ms:        float


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _preprocess_input(features: List[float]) -> np.ndarray:
    """Convert raw feature list to model-ready numpy array."""
    x = np.array(features, dtype=np.float32).reshape(1, -1)

    if _preprocessor is not None:
        try:
            if hasattr(_preprocessor, "transform"):
                x = _preprocessor.transform(x).astype(np.float32)
            elif hasattr(_preprocessor, "scaler"):
                x = _preprocessor.scaler.transform(x).astype(np.float32)
        except Exception:
            pass  # If scaler fails, use raw values

    return x


def _run_model(x: np.ndarray):
    """
    Run inference. Returns (proba, gate_dict|None).

    Keras 3 returns DICT outputs for multi-output models.
    Older behaviour (list) is also handled for compatibility.
    Single-output models return a plain array.
    """
    if _IS_FLOW_FUSION:
        import tensorflow as tf
        out = _model.predict(x, verbose=0)

        # Bug 7 fix: Keras 3 returns dict for named multi-output models.
        if isinstance(out, dict):
            proba     = out["output"][0]                           # (n_classes,)
            gate_keys = sorted(k for k in out if k.startswith("gate_"))
            gates     = [float(out[k][0][0]) for k in gate_keys]  # scalar per branch
            names     = [k.replace("gate_", "") for k in gate_keys]
            gate_dict = dict(zip(names, gates)) if gates else None
        elif isinstance(out, list):
            proba     = out[0][0]                          # (n_classes,)
            gates     = [float(g[0][0]) for g in out[1:]]
            n         = len(gates)
            names     = _FLOW_FUSION_BRANCH_NAMES[:n]
            gate_dict = dict(zip(names, gates)) if gates else None
        else:
            # Single-output FlowFusion (branch-only configs)
            proba     = out[0]
            gate_dict = None
        return proba, gate_dict
    else:
        proba = _model.predict_proba(x)[0]
        return proba, None


def _check_ready():
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. Run `python scripts/train_xgboost.py` "
                "first to generate models/xgb_best.joblib."
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Dashboard"])
async def root():
    """Serve the interactive DDoS Detection & Simulation Dashboard HTML."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "status":       "ok",
        "service":      "DDoS Detection API",
        "version":      _VERSION,
        "model":        _MODEL_NAME,
        "model_loaded": _model is not None,
        "is_flowfusion": _IS_FLOW_FUSION,
    }


@app.get("/health", tags=["Health"])
async def health():
    """Health check endpoint."""
    return {
        "status":       "ok",
        "service":      "DDoS Detection API",
        "version":      _VERSION,
        "model":        _MODEL_NAME,
        "model_loaded": _model is not None,
        "is_flowfusion": _IS_FLOW_FUSION,
    }


@app.get("/info", tags=["Info"])
async def info():
    """Return model metadata: feature count, class names, version."""
    classes      = list(_label_classes) if _label_classes is not None else []
    feature_count= len(_feature_names)  if _feature_names is not None else 0
    return {
        "version":       _VERSION,
        "model":         _MODEL_NAME,
        "num_classes":   len(classes),
        "classes":       classes,
        "num_features":  feature_count,
        "feature_names": list(_feature_names) if _feature_names is not None else [],
    }


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(sample: FlowSample):
    """
    Predict the DDoS class for a single network flow.

    Body: `{"features": [f1, f2, ..., fn]}`

    When the loaded model is FlowFusion, gate values are computed internally
    but not returned here — use `/explain` to retrieve them.
    """
    _check_ready()

    t0    = time.perf_counter()
    X     = _preprocess_input(sample.features)
    proba, _ = _run_model(X)
    idx   = int(np.argmax(proba))

    classes = (
        list(_label_classes) if _label_classes is not None
        else [str(i) for i in range(len(proba))]
    )
    latency = (time.perf_counter() - t0) * 1000

    return PredictResponse(
        predicted_class   = classes[idx],
        predicted_index   = idx,
        confidence        = float(proba[idx]),
        all_probabilities = {cls: float(p) for cls, p in zip(classes, proba)},
        model_type        = _MODEL_NAME,
        latency_ms        = round(latency, 3),
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(sample: FlowSample, top_k: int = 10):
    """
    Predict + return feature attributions and (for FlowFusion) branch gate values.

    Body: `{"features": [f1, f2, ..., fn]}`
    Query param: `?top_k=10` (number of top features to return)

    - Tree models (XGBoost, LightGBM): returns TreeSHAP attributions per feature.
    - Deep models (MLP, FT-Transformer): returns Integrated Gradients attributions.
    - FlowFusion: returns IG attributions + `gate_values` — 3 scalar values showing
      which branch the model trusted most for this specific flow.
      Example: {"tabular": 0.71, "spatial": 0.12, "structural": 0.17}
    """
    _check_ready()

    t0 = time.perf_counter()
    X  = _preprocess_input(sample.features)

    proba, gate_dict = _run_model(X)
    idx     = int(np.argmax(proba))
    classes = (
        list(_label_classes) if _label_classes is not None
        else [str(i) for i in range(len(proba))]
    )

    fnames = (
        list(_feature_names) if _feature_names is not None
        else [f"feature_{i}" for i in range(X.shape[1])]
    )

    top_features = []

    if _IS_FLOW_FUSION:
        # Integrated Gradients for deep models
        try:
            import tensorflow as tf
            with tf.GradientTape() as tape:
                x_tf = tf.constant(X)
                tape.watch(x_tf)
                out = _model(x_tf, training=False)
                logits = out[0] if isinstance(out, list) else out
                target = logits[:, idx]
            grads     = tape.gradient(target, x_tf).numpy()[0]    # (n_features,)
            attrs     = np.abs(grads * X[0])
            sorted_i  = np.argsort(attrs)[::-1][:top_k]
            top_features = [
                {"feature": fnames[i], "attribution": float(attrs[i])}
                for i in sorted_i
            ]
        except Exception as e:
            top_features = [{"feature": f"ig_error: {str(e)[:80]}", "attribution": 0.0}]
    else:
        # TreeSHAP for tree-based models
        try:
            import shap
            explainer = shap.TreeExplainer(_model)
            shap_vals = explainer.shap_values(X)
            sv_class  = shap_vals[idx][0] if isinstance(shap_vals, list) else shap_vals[0]
            sorted_i  = np.argsort(np.abs(sv_class))[::-1][:top_k]
            top_features = [
                {
                    "feature":    fnames[i],
                    "shap_value": float(sv_class[i]),
                    "abs_shap":   float(abs(sv_class[i])),
                }
                for i in sorted_i
            ]
        except ImportError:
            top_features = [{"feature": "shap_not_installed", "shap_value": 0.0, "abs_shap": 0.0}]
        except Exception as e:
            top_features = [{"feature": f"shap_error: {str(e)[:80]}", "shap_value": 0.0, "abs_shap": 0.0}]

    latency = (time.perf_counter() - t0) * 1000

    return ExplainResponse(
        predicted_class = classes[idx],
        confidence      = float(proba[idx]),
        model_type      = _MODEL_NAME,
        top_features    = top_features,
        gate_values     = gate_dict,    # None for non-FlowFusion models
        latency_ms      = round(latency, 3),
    )
