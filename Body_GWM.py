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


#TURBINE SPECIFICATIONS as a CLASS 

class TurbineModel:
    def __init__(
        self,
        name,
        rotor_diameter,
        hub_height,
        rated_power_kw,
        cut_in_speed,
        rated_speed,
        cut_out_speed,
        power_curve_path,
        thrust_coefficient=0.8,
    ):
        self.name = name
        self.rotor_diameter = rotor_diameter
        self.hub_height = hub_height
        self.rated_power_kw = rated_power_kw
        self.cut_in_speed = cut_in_speed
        self.rated_speed = rated_speed
        self.cut_out_speed = cut_out_speed
        self.thrust_coefficient = thrust_coefficient

        self.power_curve_path = power_curve_path #idk where to take this from

# MAIN Turbine Model for Vestas V110-2.0 MW
vestas = TurbineModel(
    name="Vestas V110-2.0 MW",
    rotor_diameter=110.0,
    hub_height=95.0,
    rated_power_kw=2000.0,
    cut_in_speed=3.0,
    rated_speed=11.5,
    cut_out_speed=20.0,
    power_curve_path="UNKNOWN",  # Placeholder for the actual power curve path
    thrust_coefficient=0.8,
)

"""
Turbine specifications: Vestas V110-2.0 MW
Source (verified datasheet values):
- Rated power: 2000 kW
- Cut-in wind speed: 3 m/s
- Rated wind speed: 11.5 m/s
- Cut-out wind speed: 20 m/s
- Rotor diameter: 110 m
- Hub height option used here: 95 m
"""


#MISSING POWER CURVE FUNCTION

