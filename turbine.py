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
        ct_curve_path,
        thrust_coefficient=0.8,
        **kwargs,
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
        self.ct_curve_path = ct_curve_path

        power_curve_data = pd.read_csv(power_curve_path)  # Load power curve data from CSV
        self.power_wind_speeds = power_curve_data['wind_speed_m_s'].values  # Wind speeds from the CSV
        self.power_output_kw = power_curve_data['power_kw'].values  # Corresponding power

        if self.power_wind_speeds.max() < self.cut_out_speed:
            raise ValueError("Power curve data does not cover the cut-out wind speed. Please provide a complete power curve.")

        ct_curve_data = pd.read_csv(ct_curve_path)  # Load thrust coefficient data from CSV
        self.ct_wind_speeds = ct_curve_data['wind_speed_m_s'].values  # Wind speeds from the CSV
        self.ct_values = ct_curve_data['thrust_coefficient'].values  # Corresponding thrust coefficients
        self.thrust_coefficient = float(np.interp(self.rated_speed, self.ct_wind_speeds, self.ct_values, left=self.ct_values[0], right=self.ct_values[-1]))

        for key, value in kwargs.items():
            setattr(self, key, value)


    @property # allows you to access functions as variables
    def rotor_area(self):
         return np.pi * (self.rotor_diameter / 2) ** 2
    
    #now interpreting the power curve
    def power_output(self, wind_speed):
        power = np.interp(wind_speed, self.power_wind_speeds, self.power_output_kw, left=0, right=0)  #interpolation and left and right values for out of bounds wind speeds
        
        power = np.where(wind_speed >= self.cut_out_speed, 0.0, power)
        return power

    def ct_at(self, wind_speed):
        """Return the interpolated thrust coefficient at the given wind speed."""
        ct = np.interp(
            wind_speed,
            self.ct_wind_speeds,
            self.ct_values,
            left=self.ct_values[0],
            right=self.ct_values[-1],
        )
        return np.clip(ct, 0.0, 1.0)

    def thrust_force(self, wind_speed, air_density=1.225):
        """Return aerodynamic thrust force for the turbine at a given effective wind speed.

        The force is zero outside the operational range [cut-in, cut-out).
        """
        U = np.asarray(wind_speed, dtype=float)
        Ct = self.ct_at(U)
        force = 0.5 * air_density * self.rotor_area * Ct * U ** 2

        outside_operating = (U < self.cut_in_speed) | (U >= self.cut_out_speed)
        if np.isscalar(U):
            return 0.0 if outside_operating else float(force)
        force[outside_operating] = 0.0
        return force
    
power_curve_path = Path(__file__).resolve().parent / "vestas_v110_actual_power_curve_table.csv"  # Placeholder for the actual power curve CSV file path
ct_curve_path = Path(__file__).resolve().parent / "vestas_v110_2mw_ct_curve.csv"  # Placeholder for the actual thrust coefficient CSV file path

if not power_curve_path.exists():
    raise FileNotFoundError(f"Power curve CSV file not found at {power_curve_path}. Please provide the correct path to the power curve data.")        

if not ct_curve_path.exists():
    raise FileNotFoundError(f"Thrust coefficient CSV file not found at {ct_curve_path}. Please provide the correct path to the thrust coefficient data.")        

# MAIN Turbine Model for Vestas V110-2.0 MW
vestas = TurbineModel(
    name="Vestas V110-2.0 MW",
    rotor_diameter=110.0,
    hub_height=95.0,
    rated_power_kw=2000.0,
    cut_in_speed=3.0,

    # Additional turbine specifications
    #sourced from Vestas V110-2.0 MW Platform brochure and general specifications

    re_cut_in_speed=18.0,
    wind_class="IEC III A",
    iec_reference_wind_speed=37.5,
    iec_reference_turbulence=0.16,
    min_operating_temperature=-20.0,
    max_operating_temperature=45.0,
    low_temp_restart = -19.0,
    high_temp_restart = 44.0,

 
    rated_speed=11.5,
    cut_out_speed=20.0,
    power_curve_path= power_curve_path,  #sourced from thewindpower.net
    ct_curve_path= ct_curve_path #thrust coefficient curve sourced from Muttenz Hardwald Wind Assessment
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
 
