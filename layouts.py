"""Baseline layouts and diverse layout-family seeds for the GA."""

import numpy as np

from turbine import vestas

D = vestas.rotor_diameter
SPACING = 5 * D
#GRID_SIZE = 5
GRID_ROW = 6
GRID_COLUMN = 6

def grid_layout():
    return [
        (row * SPACING, column * SPACING)
        for row in range(GRID_ROW)
        for column in range(GRID_COLUMN)
    ]


def staggered_layout():
    return [
        (
            row * SPACING,
            column * SPACING + (SPACING / 2 if row % 2 else 0.0),
        )
        for row in range(GRID_ROW)
        for column in range(GRID_COLUMN)
    ]


def rotate_layout(positions, angle_deg):
    layout = np.asarray(positions, dtype=float)
    centre = layout.mean(axis=0)
    theta = np.radians(angle_deg)
    rotation = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    return (layout - centre) @ rotation.T + centre


def hexagonal_layout(spacing=4.5 * D):
    return [
        (
            column * spacing + (0.5 * spacing if row % 2 else 0.0),
            row * spacing * np.sqrt(3.0) / 2.0,
        )
        for row in range(GRID_ROW)
        for column in range(GRID_COLUMN)
    ]


def offset_rows_layout(offset=1.25 * D):
    return [
        (column * SPACING + row * offset, row * SPACING)
        for row in range(GRID_ROW)
        for column in range(GRID_COLUMN)
    ]


def skewed_staggered_layout(skew=0.75 * D):
    return [
        (
            column * SPACING + row * skew,
            row * SPACING + (0.5 * SPACING if column % 2 else 0.0),
        )
        for row in range(GRID_ROW)
        for column in range(GRID_COLUMN)
    ]


def irregular_layout(rng):
    layout = np.asarray(staggered_layout(), dtype=float)
    return layout + rng.normal(0.0, 0.6 * D, layout.shape)


def layout_families(rng):
    staggered = np.asarray(staggered_layout(), dtype=float)
    return [
        staggered,
        rotate_layout(staggered, 15.0),
        np.asarray(hexagonal_layout()),
        np.asarray(offset_rows_layout()),
        np.asarray(skewed_staggered_layout()),
        irregular_layout(rng),
    ]
