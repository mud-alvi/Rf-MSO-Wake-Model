"""
Generates two candidate 5x5 turbine layouts to compare, per the feedback
doc's instruction: prove the method works on a small array rather than
building a full-scale GA right now.

Layout 1: Standard grid, 5D x 5D spacing (within your paper's cited
          spacing range of 3-5D crosswind / 6-10D downstream -- 5D is a
          reasonable baseline for both directions here).
Layout 2: Staggered, offset by half the crosswind spacing every other row,
          approximating alignment with a prevailing wind direction.
"""

import numpy as np
from turbine import vestas

D = vestas.rotor_diameter
SPACING = 5 * D  # 5 rotor diameters, matches your paper's spacing table
GRID_SIZE = 5    # 5x5 = 25 turbines


def grid_layout():
    positions = []
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = i * SPACING
            y = j * SPACING
            positions.append((x, y))
    return positions


def staggered_layout():
    positions = []
    for i in range(GRID_SIZE):
        row_offset = (SPACING / 2) if (i % 2 == 1) else 0
        for j in range(GRID_SIZE):
            x = i * SPACING
            y = j * SPACING + row_offset
            positions.append((x, y))
    return positions
