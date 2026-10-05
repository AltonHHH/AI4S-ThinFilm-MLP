import numpy as np, torch
from torch import nn
# See the accompanying report for the exact parameter values.
# TMM: Air/H/L/H/L/Glass, nH=2.30, nL=1.45, ns=1.52
# wavelength: 400–800 nm, 10 nm step
# student-specific: target=600 nm, seed=270177, design_seed=270178
