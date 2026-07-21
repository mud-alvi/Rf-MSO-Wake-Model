"""Generate 5x5 grid and staggered layouts for the wake experiment.

Coordinates returned by this module are geographic: x is east and y is
north.  When a prevailing meteorological wind direction is supplied, the
base layout's row axis is rotated to point downwind.  Both candidate layouts
therefore use the same site orientation and can be compared fairly.
"""

import numpy as np

from turbine import vestas


D = vestas.rotor_diameter
SPACING = 5 * D
GRID_SIZE = 5


def rotate_layout_to_wind(positions, wind_from_direction):
    """Align a base layout's +x row axis with the prevailing wind flow.

    ``wind_from_direction`` uses the meteorological convention: 0 degrees
    means wind from north, 90 from east, 180 from south, and 270 from west.
    The returned coordinates use x=east and y=north.
    """
    positions = np.asarray(positions, dtype=float)
    direction = np.radians(float(wind_from_direction) % 360)

    # Unit vectors in geographic (east, north) coordinates.
    downwind = np.array([-np.sin(direction), -np.cos(direction)])
    crosswind = np.array([np.cos(direction), -np.sin(direction)])

    rotated = (
        positions[:, [0]] * downwind
        + positions[:, [1]] * crosswind
    )

    # Translation does not affect wake distances. Keeping the minimum at zero
    # makes the layout graph easier to read.
    rotated -= rotated.min(axis=0)
    return list(map(tuple, rotated))


def _orient(positions, prevailing_direction_deg):
    if prevailing_direction_deg is None:
        return positions
    return rotate_layout_to_wind(positions, prevailing_direction_deg)


def grid_layout(prevailing_direction_deg=None):
    positions = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            positions.append((i * SPACING, j * SPACING))
    return _orient(positions, prevailing_direction_deg)


def staggered_layout(prevailing_direction_deg=None):
    positions = []
    for i in range(GRID_SIZE):
        row_offset = (SPACING / 2) if (i % 2 == 1) else 0
        for j in range(GRID_SIZE):
            positions.append((i * SPACING, j * SPACING + row_offset))
    return _orient(positions, prevailing_direction_deg)
