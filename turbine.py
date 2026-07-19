import numpy as np
import pandas as pd
from pathlib import Path



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

        self.power_curve_path = power_curve_path 

        power_curve_data = pd.read_csv(power_curve_path)  # Load power curve data from CSV
        self.power_wind_speeds = power_curve_data['wind_speed_m_s'].values  # Wind speeds from the CSV
        self.power_output_kw = power_curve_data['power_kw'].values  # Corresponding power

        if self.power_wind_speeds.max() < self.cut_out_speed:
            raise ValueError("Power curve data does not cover the cut-out wind speed. Please provide a complete power curve.")

    @property # allows you to access functions as variables
    def rotor_area(self):
         return np.pi * (self.rotor_diameter / 2) ** 2
    
    #now interpreting the power curve
    def power_output(self, wind_speed):
        power = np.interp(wind_speed, self.power_wind_speeds, self.power_output_kw, left=0, right=0)  #interpolation and left and right values for out of bounds wind speeds
        
        power = np.where(wind_speed >= self.cut_out_speed,0.0, power)
        return power
    
power_curve_path = Path(__file__).resolve().parent / "vestas_v110_actual_power_curve_table.csv"  # Placeholder for the actual power curve CSV file path

if not power_curve_path.exists():
    raise FileNotFoundError(f"Power curve CSV file not found at {power_curve_path}. Please provide the correct path to the power curve data.")        

# MAIN Turbine Model for Vestas V110-2.0 MW
vestas = TurbineModel(
    name="Vestas V110-2.0 MW",
    rotor_diameter=110.0,
    hub_height=95.0,
    rated_power_kw=2000.0,
    cut_in_speed=3.0,
    rated_speed=11.5,
    cut_out_speed=20.0,
    power_curve_path= power_curve_path,  #sourced from thewindpower.net
    thrust_coefficient=0.8
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




#TEST
if __name__ == "__main__":
    print(round(vestas.rotor_area, 2))  #
    print(vestas.power_output(5))  # Regular test
    print(vestas.power_output(20)) 
    print(vestas.power_output(0))  # should be 0
 
