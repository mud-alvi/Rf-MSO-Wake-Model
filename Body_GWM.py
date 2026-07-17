import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import windrose

#sublibrary imports

from scipy.special import erf #for error function calculations
from scipy.optimize import curve_fit #for fitting functions
from scipy.optimize import minimize #for optimization
from scipy.optimize import brentq #for root finding
from scipy.spatial.distance import cdist #for distance calculations
from scipy.stats import weibull_min #for weibull distribution (probability density function)
from scipy.interpolate import interpolate #for interpolation (estimating values between known data points)
from scipy.integrate import quad #for integration

value = erf(1) 
print(value)

print("Hello, World!")