def to_gy(dose_in_mev_per_g: float) -> float:
    """Convert dose from MeV/g to Gy.

    1 MeV/g = 1.60218e-10 Gy

    Args:
        dose_in_mev_per_g (float): Dose in MeV/g.

    Returns:
        float: Dose in Gy.
    """
    return dose_in_mev_per_g * 1.60218e-10