"""Pipeline configuration: one validated, serializable object per run.

Every tunable number of the processing chain lives here with its default and
the experiment that set it, so a run is fully described by its YAML file (see
``configs/pipeline/default.yaml``) and the config hash recorded with it.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sentinel.detection.cfar import CFAR_THRESHOLD_PFA_1E2


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DetectionConfig(_Strict):
    method: Literal["cfar", "cusum", "glrt"] = "cfar"
    threshold: float = Field(
        CFAR_THRESHOLD_PFA_1E2,
        description="Window score threshold; the default is CFAR calibrated to "
        "1e-2 false alarms per 64 s window (E1). Recalibrate for other methods.",
    )
    window_s: float = Field(64.0, gt=0, description="Pixel window length, s.")


class ClassificationConfig(_Strict):
    artifact: Path | None = Field(
        None,
        description="Exported classifier directory; None reports detections unclassified.",
    )
    backend: Literal["torch", "onnx"] = Field(
        "torch",
        description="Inference runtime; 'onnx' needs `sentinel export-onnx` "
        "and the onnx extra.",
    )
    onset_range_s: tuple[float, float] = Field(
        (2.0, 40.0),
        description="Onset positions within the window the classifier was "
        "trained on; class evidence outside it is not fused (E6).",
    )
    reject_background: bool = Field(
        True, description="Drop detections the classifier labels background (glints)."
    )

    @model_validator(mode="after")
    def _ordered(self) -> "ClassificationConfig":
        low, high = self.onset_range_s
        if not 0.0 <= low < high:
            raise ValueError("onset_range_s must satisfy 0 <= low < high")
        return self


class RFConfig(_Strict):
    clock_bias_ns: float = Field(0.0, ge=0, description="Receiver clock-bias level.")
    survey_m: float = Field(0.0, ge=0, description="Receiver survey error per axis.")
    lo_offset_hz: float = Field(0.0, ge=0, description="Local-oscillator offset level.")
    fit_probability: float = Field(
        0.999, gt=0, lt=1, description="χ² fit level a fix must pass to be fused."
    )
    late_fixes: Literal["extrapolate", "drop"] = Field(
        "extrapolate",
        description="Fixes older than the frame: propagate with their FDOA "
        "velocity (E7: lossless to 2 s) or drop.",
    )

    @property
    def has_systematics(self) -> bool:
        return max(self.clock_bias_ns, self.survey_m, self.lo_offset_hz) > 0


class OPIRConfig(_Strict):
    angle_std_rad: float = Field(
        10e-6, gt=0, description="Line-of-sight error per axis."
    )
    stereo_gate_probability: float = Field(0.99, gt=0, lt=1)
    reject_ambiguous: bool = Field(
        True, description="Defer ambiguous stereo pairs to line-of-sight updates (E5)."
    )


class TrackingConfig(_Strict):
    gate_probability: float = Field(0.99, gt=0, lt=1)
    confirm_hits: int = Field(3, ge=1, description="M of M-of-N confirmation (E5).")
    confirm_window: int = Field(5, ge=1, description="N of M-of-N confirmation.")
    max_coast_s: float = Field(10.0, ge=0)
    initial_velocity_std: float = Field(300.0, gt=0)
    class_weight: float = Field(0.3, gt=0, le=1, description="Class pooling weight.")
    merge_probability: float | None = Field(0.9, gt=0, lt=1)
    imm: bool = True
    imm_noise: tuple[float, float] = (25.0, 1_600.0)
    imm_sojourn_s: tuple[float, float] = (60.0, 10.0)

    @model_validator(mode="after")
    def _window(self) -> "TrackingConfig":
        if self.confirm_hits > self.confirm_window:
            raise ValueError("confirm_hits must not exceed confirm_window")
        return self


class PipelineConfig(_Strict):
    """Configuration of :class:`~sentinel.pipeline.runner.SentinelPipeline`."""

    detection: DetectionConfig = DetectionConfig()
    classification: ClassificationConfig = ClassificationConfig()
    rf: RFConfig = RFConfig()
    opir: OPIRConfig = OPIRConfig()
    tracking: TrackingConfig = TrackingConfig()

    def sha256(self) -> str:
        """Hash of the canonical JSON form (identifies the configuration)."""
        canonical = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Read and validate a pipeline YAML file. A relative ``artifact`` path is
    interpreted relative to the current working directory (the repository
    root when run through ``make`` or the CLI)."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    return PipelineConfig.model_validate(data)
