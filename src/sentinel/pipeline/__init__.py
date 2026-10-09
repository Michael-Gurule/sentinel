"""The SENTINEL processing chain: configuration, stage interfaces, and runner."""

from sentinel.pipeline.config import (
    ClassificationConfig,
    DetectionConfig,
    OPIRConfig,
    PipelineConfig,
    RFConfig,
    TrackingConfig,
    load_pipeline_config,
)
from sentinel.pipeline.runner import (
    FrameResult,
    OPIRObservation,
    OPIRReport,
    SensorFrame,
    SentinelPipeline,
    track_to_dict,
)
from sentinel.pipeline.scenario import ScenarioRun, frames_from_scenario, run_scenario
from sentinel.pipeline.stages import (
    Classifier,
    RFGeolocator,
    RFObservation,
    TDOAFDOAGeolocator,
    Tracker,
)

__all__ = [
    "ClassificationConfig",
    "Classifier",
    "DetectionConfig",
    "FrameResult",
    "OPIRConfig",
    "OPIRObservation",
    "OPIRReport",
    "PipelineConfig",
    "RFConfig",
    "RFGeolocator",
    "RFObservation",
    "ScenarioRun",
    "SensorFrame",
    "SentinelPipeline",
    "TDOAFDOAGeolocator",
    "Tracker",
    "TrackingConfig",
    "frames_from_scenario",
    "load_pipeline_config",
    "run_scenario",
    "track_to_dict",
]
