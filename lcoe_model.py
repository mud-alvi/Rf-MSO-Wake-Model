import math


TURBINE_POWER = 2000
TURBINES = 60

CAPEX_TOTAL_PER_KW = 1968
PLANT_CAPACITY_KW =TURBINE_POWER * TURBINES  # Example plant capacity
BASE_CAPEX_USD = CAPEX_TOTAL_PER_KW * PLANT_CAPACITY_KW  #for 25: 98,400,000
OPEX_PER_KW_YEAR = 43
ANNUAL_OPEX_USD = OPEX_PER_KW_YEAR * PLANT_CAPACITY_KW  #for 25: 2,150,000
PROJECT_LIFE_YEARS = 25
REAL_WACC = 0.0366
REAL_FCR = 0.065
CABLE_COST_PER_M = 324.68
ROAD_COST_PER_M = 88
SUBSTATION = (0.0, 1237.5)
ROAD_ENTRY = (0.0, 1237.5)


def distance(point_1, point_2):
    return math.hypot(point_2[0] - point_1[0], point_2[1] - point_1[1])


def minimum_spanning_tree_length(points):
    points = [tuple(map(float, point)) for point in points]
    if len(points) <= 1:
        return 0.0
    connected = [False] * len(points)
    shortest = [math.inf] * len(points)
    shortest[0] = 0.0
    total = 0.0
    for _ in points:
        next_point = min(
            (i for i in range(len(points)) if not connected[i]),
            key=shortest.__getitem__,
        )
        connected[next_point] = True
        total += shortest[next_point]
        for other in range(len(points)):
            if not connected[other]:
                shortest[other] = min(
                    shortest[other], distance(points[next_point], points[other])
                )
    return total


def calculate_lcoe_details(layout, annual_aep_mwh):
    if layout is None or len(layout) == 0:
        raise ValueError("The layout cannot be empty.")
    if annual_aep_mwh <= 0:
        raise ValueError("Annual AEP must be greater than zero.")

    turbines = [tuple(map(float, position)) for position in layout]
    cable_length = minimum_spanning_tree_length([SUBSTATION, *turbines])
    road_length = minimum_spanning_tree_length([ROAD_ENTRY, *turbines])
    cable_cost = cable_length * CABLE_COST_PER_M
    road_cost = road_length * ROAD_COST_PER_M
    total_capex = BASE_CAPEX_USD + cable_cost + road_cost
    annual_cost = REAL_FCR * total_capex + ANNUAL_OPEX_USD
    return {
        "lcoe": annual_cost / annual_aep_mwh,
        "cable_length_m": cable_length,
        "road_length_m": road_length,
        "cable_cost_usd": cable_cost,
        "road_cost_usd": road_cost,
        "total_capex": total_capex,
        "annual_project_cost_usd": annual_cost,
    }


def calculate_lcoe(layout, annual_aep_mwh):
    return calculate_lcoe_details(layout, annual_aep_mwh)["lcoe"]


if __name__ == "__main__":
    from layouts import staggered_layout
    from main import calculate_aep, load_real_weather_data

    weather = load_real_weather_data()
    layout = staggered_layout()
    aep, _ = calculate_aep(
        layout,
        weather["wind_speed_m_s"],
        weather["wind_direction_deg"],
        weather["air_density_kg_m3"],
    )
    print(f"LCOE: ${calculate_lcoe(layout, aep):,.2f}/MWh")
