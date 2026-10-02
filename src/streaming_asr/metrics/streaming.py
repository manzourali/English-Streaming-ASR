def real_time_factor(processing_seconds: float, audio_seconds: float) -> float:
    if audio_seconds <= 0:
        raise ValueError("audio_seconds must be positive")
    return processing_seconds / audio_seconds


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

