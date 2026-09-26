"""Shared exceptions for fictional document generation."""


class StrictInterVariantUniquenessInfeasible(RuntimeError):
    """Raised when strict inter-variant numeric/temporal uniqueness has no solution."""
