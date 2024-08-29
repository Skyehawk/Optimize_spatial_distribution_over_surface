# Author: Skye Leake
# Last Updated: 08-28-2024
#
# Provided under the MIT OpenSource License

# plot_functions.py


from __future__ import annotations

from typing import Optional  # Deprecated in Python 3.9, need to swap to collections.abc

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

# from shapely.geometry import Polygon as ShapelyPolygon


def plot_generated_surface(surface: np.ndarray, valid_mask: np.ndarray) -> None:
    """
    Create a Matplotlib figure of the input arrays using imshow()

    Arguments:
    surface        -- Numpy ndarray - representing the surface (2 dimesional)
    valid_mask -- Numpy ndarray - representing the valid areas mask (2 dimentional)

    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    c1 = axs[0].imshow(
        surface, cmap="terrain", origin="lower"
    )  # imshow places orgin in upper left (image orgin), we want lower left (cartesian crds)

    axs[0].set_title("Digital Elevation Model (DEM)")
    fig.colorbar(c1, ax=axs[0], orientation="vertical", shrink=0.6)

    c2 = axs[1].imshow(
        valid_mask, cmap="Greens", alpha=0.5, origin="lower"
    )  # same as above, but for the valid areas of the domain

    axs[1].set_title("Valid Areas Mask")
    fig.colorbar(c2, ax=axs[1], orientation="vertical", shrink=0.6)

    plt.show()


def plot_observation_points_with_polygons_3d(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    observation_points: list[list[int | float]],
    fixed_points: list[list[int | float]],
    observation_polygons: list[ShapelyPolygon],
    fixed_polygons: list[ShapelyPolygon],
    annotation: Optional[str] = None,
    ax: Optional[Axes] = None,
) -> None:
    """
    Create a Matplotlib figure with visible area polygons
    """
    # Make some sort of figure with some defaults if none provided
    # fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.cla()

    # Create grid coordinates for pcolormesh
    x, y = np.meshgrid(
        np.linspace(0, surface.shape[1], surface.shape[1] + 1),
        np.linspace(0, surface.shape[0], surface.shape[0] + 1),
    )

    # Plot surface as pcolormesh
    ax.pcolormesh(x, y, surface, cmap="terrain", shading="auto")
    ax.set_title("Digital Elevation Model (surface)")
    # ax.colorbar(label="Elevation", shrink=0.6)

    # Plot observation points
    ax.scatter(
        [p[1] for p in observation_points],
        [p[0] for p in observation_points],
        c="red",
        label="Observation Points",
    )

    # Plot fixed points
    ax.scatter(
        [p[1] for p in fixed_points],
        [p[0] for p in fixed_points],
        c="blue",
        label="Fixed Points",
    )
    ax.legend()

    # Create grid coordinates for valid_mask
    valid_x, valid_y = np.meshgrid(
        np.arange(valid_mask.shape[1] + 1), np.arange(valid_mask.shape[0] + 1)
    )

    # Overlay filled contour with hatching for valid and invalid areas based on the valid mask
    ax.contourf(
        valid_x[:-1, :-1],
        valid_y[:-1, :-1],
        valid_mask,
        colors="gray",
        levels=[0, 0.5],
        hatches=["", "\\"],
        alpha=0.5,
    )

    # Plot polygons for visible areas
    colors = [
        "blue",
        "green",
        "orange",
        "cyan",
        "magenta",
        "plum",
        "goldenrod",
        "palegreen",
    ]  # Unique colors for each observation point

    # Plot observation polygons
    for idx, polygon in enumerate(observation_polygons):
        if polygon.is_empty:
            continue
        x, y = polygon.exterior.xy
        ax.fill(
            y,
            x,
            color=colors[idx % len(colors)],
            linewidth=2,
            alpha=0.3,
            label=f"Observation Polygon {idx+1}",
        )

    # Plot fixed polygons with different colors
    fixed_colors = [
        "brown",
        "purple",
        "pink",
        "lime",
        "navy",
        "teal",
        "grey",
        "olive",
    ]  # Unique colors for each fixed point
    for idx, polygon in enumerate(fixed_polygons):
        if polygon.is_empty:
            continue
        x, y = polygon.exterior.xy
        ax.fill(
            y,
            x,
            color=fixed_colors[idx % len(fixed_colors)],
            linewidth=2,
            alpha=0.3,
            label=f"Fixed Polygon {idx+1}",
        )

    if isinstance(annotation, str):
        # Add annotation at the bottom center, used for callback functionality to
        # see progress as optimizer runs
        ax.annotate(
            annotation,
            xy=(0.5, 0),
            xytext=(0.5, -0.15),
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1),
        )

    fig.gca().set_aspect("equal", adjustable="box")  # Ensure square pixels
    fig.tight_layout()


# EOF
