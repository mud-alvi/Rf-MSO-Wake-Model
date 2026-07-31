import numpy as np
import pandas as pd
from pathlib import Path


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

        power_curve = pd.read_csv(power_curve_path)
        self.power_wind_speeds = power_curve["wind_speed_m_s"].values
        self.power_output_kw = power_curve["power_kw"].values
        if self.power_wind_speeds.max() < self.cut_out_speed:
            raise ValueError("Power curve data does not cover the cut-out wind speed.")

        ct_curve = pd.read_csv(ct_curve_path)
        self.ct_wind_speeds = ct_curve["wind_speed_m_s"].values
        self.ct_values = ct_curve["thrust_coefficient"].values
        self.thrust_coefficient = float(self.ct_at(self.rated_speed))

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def rotor_area(self):
        return np.pi * (self.rotor_diameter / 2) ** 2

    def power_output(self, wind_speed):
        power = np.interp(
            wind_speed,
            self.power_wind_speeds,
            self.power_output_kw,
            left=0,
            right=0,
        )
        return np.where(np.asarray(wind_speed) >= self.cut_out_speed, 0.0, power)

    def ct_at(self, wind_speed):
        """Return a stable interpolated thrust coefficient."""
        ct = np.interp(
            wind_speed,
            self.ct_wind_speeds,
            self.ct_values,
            left=self.ct_values[0],
            right=self.ct_values[-1],
        )
        return np.clip(ct, 0.0, 0.999)

    def thrust_force(self, wind_speed, air_density=1.225):
        U = np.asarray(wind_speed, dtype=float)
        force = 0.5 * air_density * self.rotor_area * self.ct_at(U) * U ** 2
        outside = (U < self.cut_in_speed) | (U >= self.cut_out_speed)
        return float(0.0 if outside else force) if U.ndim == 0 else np.where(outside, 0.0, force)


base_dir = Path(__file__).resolve().parent
power_curve_path = base_dir / "vestas_v110_actual_power_curve_table.csv"
ct_curve_path = base_dir / "vestas_v110_2mw_ct_curve.csv"

if not power_curve_path.exists():
    raise FileNotFoundError(f"Power curve CSV file not found at {power_curve_path}.")
if not ct_curve_path.exists():
    raise FileNotFoundError(f"Thrust coefficient CSV file not found at {ct_curve_path}.")

vestas = TurbineModel(
    name="Vestas V110-2.0 MW",
    rotor_diameter=110.0,
    hub_height=95.0,
    rated_power_kw=2000.0,
    cut_in_speed=3.0,
    rated_speed=11.5,
    cut_out_speed=20.0,
    re_cut_in_speed=18.0,
    wind_class="IEC III A",
    iec_reference_wind_speed=37.5,
    iec_reference_turbulence=0.16,
    min_operating_temperature=-20.0,
    max_operating_temperature=45.0,
    low_temp_restart=-19.0,
    high_temp_restart=44.0,
    power_curve_path=power_curve_path,
    ct_curve_path=ct_curve_path,
)


if __name__ == "__main__":
    print(round(vestas.rotor_area, 2))
    print(vestas.power_output(5))
