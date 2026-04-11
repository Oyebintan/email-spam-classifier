from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import tflite_runtime.interpreter as tflite


@dataclass
class InferenceArtifacts:
    feature_pipeline: Any
    l1_selector: Any
    label_encoder: Any
    interpreter: tflite.Interpreter   # ← replaces tf.keras.Model
    input_index: int
    output_index: int
    artifact_dir: Path


class SpamPredictor:
    def __init__(self, artifact_dir: str | None = None) -> None:
        self.art = self._load_artifacts(artifact_dir)

    def predict(self, text: str) -> Dict[str, Any]:
        text = self._normalize_input(text)

        if not text:
            return {"label": "ham", "probability": 0.0, "confidence": 0.0}

        x = self.art.feature_pipeline.transform([text])

        if hasattr(x, "toarray"):
            x = x.toarray().astype(np.float32)
        else:
            x = np.asarray(x, dtype=np.float32)

        x_selected = self.art.l1_selector.transform(x).astype(np.float32)

        # ← tflite inference (replaces model.predict)
        self.art.interpreter.set_tensor(self.art.input_index, x_selected)
        self.art.interpreter.invoke()
        proba_spam = float(self.art.interpreter.get_tensor(self.art.output_index).ravel()[0])
        proba_spam = max(0.0, min(1.0, proba_spam))

        label = "spam" if proba_spam >= 0.5 else "ham"
        confidence = proba_spam if label == "spam" else (1.0 - proba_spam)

        return {
            "label": label,
            "probability": round(proba_spam, 6),
            "confidence": round(confidence * 100, 2),
        }

    def _normalize_input(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\n", " ").replace("\r", " ")
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _load_artifacts(self, artifact_dir: str | None) -> InferenceArtifacts:
        base_dir = Path(__file__).resolve().parent
        project_root = base_dir.parent
        out_dir = Path(artifact_dir) if artifact_dir else (project_root / "outputs_dl")

        pipeline_path = out_dir / "pipeline.pkl"
        model_path = out_dir / "model.tflite"   # ← changed from model.keras

        missing_files = [str(p) for p in [pipeline_path, model_path] if not p.exists()]
        if missing_files:
            raise FileNotFoundError("Missing required artifact(s): " + ", ".join(missing_files))

        pipeline_obj = joblib.load(pipeline_path)
        feature_pipeline = pipeline_obj.get("feature_pipeline")
        l1_selector = pipeline_obj.get("l1_selector")
        label_encoder = pipeline_obj.get("label_encoder")

        if feature_pipeline is None or l1_selector is None:
            raise ValueError("pipeline.pkl is missing required preprocessing objects.")

        # ← load tflite model
        interpreter = tflite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        return InferenceArtifacts(
            feature_pipeline=feature_pipeline,
            l1_selector=l1_selector,
            label_encoder=label_encoder,
            interpreter=interpreter,
            input_index=input_index,
            output_index=output_index,
            artifact_dir=out_dir,
        )