"""Unified multi-talker pipeline; every audio sample reaches SURT."""
from .streaming import StreamingASRPipeline

OverlapPipeline = StreamingASRPipeline
