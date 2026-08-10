"""System inspection and benchmark logging utilities."""

from .inspect_outputs import inspect_workspace_outputs
from .log_benchmark import log_execution_time, record_stage_timing

__all__ = ["inspect_workspace_outputs", "record_stage_timing", "log_execution_time"]
