"""FEL controllers grouped by the feedback loop they assist.

Joint-PD FEL:
    PhaseFELController
    GaussianRBFFELController

Posture-PD / stabilizer FEL:
    PostureGaussianRBFFELController
"""

from phase_fel import PhaseFELController
from rbf_fel import GaussianRBFFELController
from posture_rbf_fel import PostureGaussianRBFFELController


__all__ = [
    'PhaseFELController',
    'GaussianRBFFELController',
    'PostureGaussianRBFFELController',
]
