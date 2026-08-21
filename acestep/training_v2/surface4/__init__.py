"""Control Surface 4: symbolic frame-aligned performance conditioning."""

from acestep.training_v2.surface4.performance_regulator import PerformanceRegulator
from acestep.training_v2.surface4.runtime_integration import attach_frozen_c25_surface4

__all__ = ["PerformanceRegulator", "attach_frozen_c25_surface4"]
