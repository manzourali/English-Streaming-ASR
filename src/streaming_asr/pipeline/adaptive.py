"""Compatibility import for the retired heuristic routing entry point."""
from .streaming import StreamingASRPipeline

AdaptiveStreamingASRPipeline = StreamingASRPipeline
AdaptivePipeline = StreamingASRPipeline
