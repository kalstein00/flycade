"""Shared CLI/Python contract for the bounded observation sampling rate."""
OBSERVATION_RATES = (0, 2, 3, 4, 5)


def validate_observation_rate(hz: int) -> None:
    if hz not in OBSERVATION_RATES:
        raise ValueError('observe-hz must be 0 (disabled) or 2..5')
