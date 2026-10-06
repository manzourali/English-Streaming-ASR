"""Explicit PEFT feasibility decisions for the selected external SURT recipe."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeftFeasibility:
    supported: bool
    status: str
    reason: str


def assess_surt_peft(exposes_torch_modules: bool = False) -> PeftFeasibility:
    if exposes_torch_modules:
        return PeftFeasibility(False, "RESEARCH DECISION PENDING", "An exact SURT recipe, module names, and checkpoint must be verified before applying LoRA.")
    return PeftFeasibility(False, "NOT TECHNICALLY SUPPORTED", "The configured Phase 7 decoder factory exposes inference only; it supplies no trainable modules, tokenizer, or loss graph for PEFT/QLoRA.")
