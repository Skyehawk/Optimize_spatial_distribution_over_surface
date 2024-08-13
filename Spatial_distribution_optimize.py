# Base imports
# Typing stuff to expidite some heavily looping code
from typing import List, Tuple

import geopandas as gpd  # v 0.14.4
import matplotlib.pyplot as plt  # Visulization - v 3.8.4
import numpy as np  # Array manipulation - v 1.26.4

# Scipy imports for spatial optimization
from scipy.ndimage import gaussian_filter  # Smoothing - v 1.13.0
from scipy.optimize import differential_evolution, minimize  # v 1.13.0

# Shapely imports for polygon handling and unions
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# We need a surface to perfom some type of spatial optimization on.
# to start we will use a square array, this is a stand-in for our
# digital elevation model

# Define a map size (keeping it small (~100)) will help with efficency
# for our initial naive approach


def generate_dem(map_size=100):

    # Generate structured DEM data
    x, y = np.meshgrid(np.linspace(-10, 10, map_size), np.linspace(-10, 10, map_size))

    # Create basin and range structure using combined functions and Gaussian filters
    dem = (
        np.sin(x) * np.cos(y) * 20
        + np.sin(0.5 * x) * np.cos(0.5 * y) * 10
        + np.sin(0.25 * x) * np.cos(0.25 * y) * 5
    )

    # Apply Gaussian filter to smooth out the DEM
    dem = gaussian_filter(dem, sigma=2)

    # Add aggressive cliffs and valleys
    cliffs = (np.tanh(2 * (x - 5)) - np.tanh(2 * (x + 5))) * 20
    valleys = -(np.tanh(2 * (y - 5)) - np.tanh(2 * (y + 5))) * 20
    dem += cliffs + valleys

    # Apply another Gaussian filter for smooth transitions
    dem = gaussian_filter(dem, sigma=1)

    # Add random noise
    dem += np.random.rand(map_size, map_size) * 5

    # Add lakes (negative values will be below sea level)
    dem -= (
        np.exp(-0.05 * ((x - 2) ** 2 + (y - 2) ** 2)) * 30
        + np.exp(-0.05 * ((x + 2) ** 2 + (y + 2) ** 2)) * 30
    )

    # Normalize DEM to be within 0 to 100 feet range
    dem = 100 * (dem - np.min(dem)) / (np.max(dem) - np.min(dem))

    # Generate structured valid areas mask using a Gaussian function
    valid_mask_base = np.exp(-0.1 * (x**2 + y**2))  # Gaussian distribution

    # Add blocks of random noise and apply Gaussian filter to create blobs
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(map_size, map_size)
    smooth_noise = gaussian_filter(
        noise, sigma=10
    )  # Adjust sigma for larger/smaller blobs of valid or invalid areas
    valid_mask = (valid_mask_base + smooth_noise) > 0.5  # Combine and threshold

    # Create grid coordinates for pcolormesh
    x, y = np.meshgrid(
        np.linspace(0, dem.shape[1], dem.shape[1] + 1),
        np.linspace(0, dem.shape[0], dem.shape[0] + 1),
    )

    return dem, valid_mask, x, y


def plot_generated_dem(dem, valid_mask, x, y):
    # Visualize the DEM and valid mask with terrain colormap and colorbars
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    c1 = axs[0].imshow(
        dem, cmap="terrain", origin="lower"
    )  # imshow places orgin in upper left (image orgin), we want lower left (cartesian crds)

    axs[0].set_title("Digital Elevation Model (DEM)")
    fig.colorbar(c1, ax=axs[0], orientation="vertical", shrink=0.6)
    c2 = axs[1].imshow(valid_mask, cmap="Greens", alpha=0.5, origin="lower")
    axs[1].set_title("Valid Areas Mask")
    fig.colorbar(c2, ax=axs[1], orientation="vertical", shrink=0.6)

    plt.show()


def get_visible_area_3d_poly(
    dem, obs_x, obs_y, obs_height, max_radius, num_transects=90
):

    polygon_points = []
    dem_shape_0, dem_shape_1 = dem.shape

    # Loop through our angles and distaces (steps).
    # This nested loop isn't the most efficent, we should probally unroll it
    for angle in np.linspace(0, 2 * np.pi, num_transects, endpoint=False):
        dx = np.cos(angle)
        dy = np.sin(angle)
        obs_elevation = dem[
            int(np.round(obs_x)), int(np.round(obs_y))
        ]  # Convert to integers

        max_distance = 0
        max_x, max_y = obs_x, obs_y

        for step in range(1, max_radius + 1):
            x = obs_x + step * dx
            y = obs_y + step * dy
            x_int = int(round(x))
            y_int = int(round(y))
            if x_int < 0 or y_int < 0 or x_int >= dem_shape_0 or y_int >= dem_shape_1:
                break

            distance_sq = (x - obs_x) ** 2 + (y - obs_y) ** 2
            target_elevation = dem[x_int, y_int]
            line_elevation = (obs_elevation + obs_height) - (
                (obs_elevation + obs_height) - target_elevation
            ) * (distance_sq / (max_radius**2))

            if target_elevation > line_elevation:
                break

            if distance_sq > max_distance:
                max_distance = distance_sq
                max_x, max_y = x_int, y_int

        polygon_points.append((max_x, max_y))  # Append the point to the polygon

    # Create a Shapely Polygon from the collected points
    polygon = ShapelyPolygon(polygon_points)
    return polygon


def get_visible_area_3d_with_obstructions(
    dem: np.ndarray,
    obs_x: int,
    obs_y: int,
    obs_height: float,
    max_radius: int,
    num_transects: int = 90,
) -> ShapelyPolygon:

    dem_size_x, dem_size_y = dem.shape
    polygon_points: List[Tuple[int, int]] = []
    obstructed_points: List[Tuple[int, int]] = []

    angles = np.linspace(0, 2 * np.pi, num_transects, endpoint=False)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    obs_elevation = dem[int(np.round(obs_x)), int(np.round(obs_y))]

    for dx, dy in zip(cos_angles, sin_angles):
        max_x, max_y = obs_x, obs_y
        transect_points: List[Tuple[int, int]] = []

        for step in range(1, max_radius + 1):
            x = obs_x + step * dx
            y = obs_y + step * dy
            x_int = int(round(x))
            y_int = int(round(y))
            if x_int < 0 or y_int < 0 or x_int >= dem_size_x or y_int >= dem_size_y:
                break
            transect_points.append((x_int, y_int))

        if not transect_points:
            continue

        distances = np.sqrt(
            (np.array([pt[0] for pt in transect_points]) - obs_x) ** 2
            + (np.array([pt[1] for pt in transect_points]) - obs_y) ** 2
        )
        back_indices = np.arange(1, max_radius + 1)[:, None]
        back_x = (
            obs_x
            + back_indices
            * (np.array([pt[0] for pt in transect_points]) - obs_x)
            / max_radius
        )
        back_y = (
            obs_y
            + back_indices
            * (np.array([pt[1] for pt in transect_points]) - obs_y)
            / max_radius
        )

        back_x_int = np.clip(np.round(back_x).astype(int), 0, dem_size_x - 1)
        back_y_int = np.clip(np.round(back_y).astype(int), 0, dem_size_y - 1)

        back_elevations = dem[back_x_int, back_y_int]

        for i, point in enumerate(transect_points):
            px, py = point
            obstructed = False

            for j in range(1, len(transect_points) + 1):
                back_elevation = back_elevations[j - 1, i]
                back_distance = distances[i]
                line_elevation_back = (obs_elevation + obs_height) - (
                    (obs_elevation + obs_height) - back_elevation
                ) * (back_distance / max_radius)

                if back_elevation > line_elevation_back:
                    obstructed = True
                    break

            if obstructed:
                obstructed_points.append(point)
            else:
                max_x, max_y = px, py

        polygon_points.append((max_x, max_y))

    # Create the main visible area polygon
    visible_polygon = ShapelyPolygon(polygon_points)

    return visible_polygon


if __name__ == "__main__":
    dem_size = 100
    plot_generated_dem(*generate_dem(dem_size))

    dem, valid, x, y = generate_dem(dem_size)

    # Observation point
    obs_x, obs_y = 50, 55
    obs_height = 10
    max_radius = 40
    num_transects = 90

    # Get the visible area with obstructions
    visible_polygon = get_visible_area_3d_poly(
        dem, obs_x, obs_y, obs_height, max_radius, num_transects
    )

    # Plot the DEM and the visible area
    fig, ax = plt.subplots(figsize=(10, 10))
    cax = ax.imshow(dem, cmap="terrain", origin="lower")
    fig.colorbar(cax, ax=ax, orientation="vertical", label="Elevation")

    # Plot the observation point
    ax.plot(obs_x, obs_y, "ro", label="Observation Point")

    # Plot the visible area polygon
    x, y = visible_polygon.exterior.xy
    ax.plot(x, y, "b-", label="Visible Area")

    # Plot the holes
    for interior in visible_polygon.interiors:
        ix, iy = zip(*interior.coords)
        ax.plot(ix, iy, "r--", label="Obstructed Area")

    # ax.legend()
    ax.set_title("Visible Area with Terrain Obstructions")
    plt.show()
