"""Phase 8 data preparation and external-recipe training lifecycle."""
from .collator import CollatedSurtBatch, SurtTrainingCollator
from .data import SurtTrainingExample, leakage_errors, select_training_condition, surt_training_example, surt_training_examples, validate_training_examples, write_surt_training_manifest
from .peft import PeftFeasibility, assess_surt_peft
from .trainer import ExternalSurtTrainingBackend, ParameterCounts, Trainer, TrainingState, load_training_backend

__all__ = ["CollatedSurtBatch", "ExternalSurtTrainingBackend", "ParameterCounts", "PeftFeasibility", "SurtTrainingCollator", "SurtTrainingExample", "Trainer", "TrainingState", "assess_surt_peft", "leakage_errors", "load_training_backend", "select_training_condition", "surt_training_example", "surt_training_examples", "validate_training_examples", "write_surt_training_manifest"]
