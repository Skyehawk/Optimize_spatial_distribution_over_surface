# Spatial_distribution_optimize.py

# Typing stuff to expidite some heavily looping code
from typing import List, Tuple

# Base imports
import geopandas as gpd  # v 0.14.4
import ipywidgets as widgets
import matplotlib.pyplot as plt  # Visulization - v 3.8.4
import numpy as np  # Array manipulation - v 1.26.4
import plotly.graph_objects as go

# Visulization libs outside of mpl
from IPython.display import HTML, clear_output, display
from ipywidgets import interact

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
    xx, yy = np.meshgrid(
        np.linspace(-10, 10, map_size), np.linspace(-10, 10, map_size)
    )  # Create basin and range structure using combined functions and Gaussian filters
    dem_to_gen = (
        np.sin(xx) * np.cos(yy) * 20
        + np.sin(0.5 * xx) * np.cos(0.5 * yy) * 10
        + np.sin(0.25 * xx) * np.cos(0.25 * yy) * 5
    )

    # Apply Gaussian filter to smooth out the DEMkl
    dem_to_gen = gaussian_filter(dem_to_gen, sigma=2)

    # Add aggressive cliffs and valleys
    cliffs = (np.tanh(2 * (xx - 5)) - np.tanh(2 * (xx + 5))) * 20
    valleys = -(np.tanh(2 * (yy - 5)) - np.tanh(2 * (yy + 5))) * 20
    dem_to_gen += cliffs + valleys

    # Apply another Gaussian filter for smooth transitions
    dem_to_gen = gaussian_filter(dem_to_gen, sigma=1)

    # Add random noise
    dem_to_gen += np.random.rand(map_size, map_size) * 5

    # Add lakes (negative values will be below sea level)
    dem_to_gen -= (
        np.exp(-0.05 * ((xx - 2) ** 2 + (yy - 2) ** 2)) * 30
        + np.exp(-0.05 * ((xx + 2) ** 2 + (yy + 2) ** 2)) * 30
    )

    # Normalize DEM to be within 0 to 100 feet range
    dem_to_gen = (
        100
        * (dem_to_gen - np.min(dem_to_gen))
        / (np.max(dem_to_gen) - np.min(dem_to_gen))
    )

    # Generate structured valid areas mask using a Gaussian function
    valid_mask_base = np.exp(-0.1 * (xx**2 + yy**2))  # Gaussian distribution

    # Add blocks of random noise and apply Gaussian filter to create blobs
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(map_size, map_size)
    smooth_noise = gaussian_filter(
        noise, sigma=10
    )  # Adjust sigma for larger/smaller blobs of valid or invalid areas
    valid_mask = (valid_mask_base + smooth_noise) > 0.5  # Combine and threshold

    # Create grid coordinates for pcolormesh
    # xx, yy = np.meshgrid(
    #    np.linspace(0, dem_to_gen.shape[1], dem_to_gen.shape[1] + 1),
    #    np.linspace(0, dem_to_gen.shape[0], dem_to_gen.shape[0] + 1),
    # )

    return dem_to_gen, valid_mask, xx, yy


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


# Callback function for visulizing the results as the optimization progresses


def callback(intermediate_result):
    # Extract current best solution
    current_solution = intermediate_result.x
    current_solution = [
        (int(current_solution[i]), int(current_solution[i + 1]))
        for i in range(0, len(current_solution), 2)
    ]

    # Grab our figure - needs to be global because the callback only supports passing an intermediate_result, no custom args
    # global fig, ax

    # Clear previous plot so we see an "updating" plot
    clear_output(wait=True)

    # Plot the current solution
    plot_observation_points_with_polygons_3d(
        dem=dem,
        valid_mask=valid_mask,
        observation_points=current_solution,
        fixed_points=fixed_points,
        observation_polygons=[],  # observation_polygons,   #pass an empty list if you don't want to plot the polys when optimizing
        fixed_polygons=[],  # fixed_polygons,
        obs_height=obs_height,
        max_radius=obs_max_radius,
        num_rays=n_rays,
        annotation=f"{current_solution}\ndifferential_evolution: f(x){intermediate_result.fun}",
        ax=ax_op,
    )

    plt.pause(0.1)  # Pause to allow the plot to update


# Optimization objective, called by visibility_optimized_points_3d
#    - Prioriries
#    - 1) Do not place on invalid areas
#    - 2) Map coverage - we desire high coverage for the given n points

# We calculate out a penalty vs shortcutting the loop in case we want "valid_areas" to be a cost surface vs a binary mask in future development


def objective(
    params,
    dem,
    obs_height,
    max_radius,
    num_rays,
    n_points,
    valid_mask,
    fixed_points,
    method,
):
    points = params.reshape(-1, 2)
    visible_area_polys = []
    penalty = 0  # Initialize penalty

    # Include fixed points in the points to be evaluated
    all_points = np.vstack((points, fixed_points))

    for i, (obs_x, obs_y) in enumerate(all_points):
        # Ensure indices are within bounds and convert to integer
        obs_x = int(np.clip(obs_x, 0, dem.shape[1] - 1))
        obs_y = int(np.clip(obs_y, 0, dem.shape[0] - 1))

        # Only check validity for observation points
        if i < len(points):
            # Check if the point is in a valid area
            if valid_mask[obs_x, obs_y] == 0:
                penalty += 1000  # Add penalty for invalid point

        if method == "3d_poly":
            visible_area = get_visible_area_3d_poly(
                dem=dem,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_height=obs_height,
                max_radius=max_radius,
                num_transects=num_rays,
            )
        elif method == "3d_poly_with_obstructions":
            visible_area = get_visible_area_3d_with_obstructions(
                dem=dem,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_height=obs_height,
                max_radius=max_radius,
                num_transects=num_rays,
            )
        else:
            raise ValueError(
                "Invalid method specified. Must be one of: '3d_poly','3d_poly_with_obstructions'"
            )

        # Handle potential invalid shapely geometries from our visibility functions
        if not visible_area.is_valid:
            visible_area = visible_area.buffer(0)

        visible_area_polys.append(visible_area)

    if visible_area_polys:
        # Calculate total visible area (map coverage) if we have any visible areas (we should)
        map_coverage = (gpd.GeoSeries(unary_union(visible_area_polys)).area).iloc[
            0
        ] / np.prod(dem.shape)
    else:
        map_coverage = 0  # return 0 if there is no valid visible_area_polys (this is an edge case, not likely we enter here)

    return -(
        map_coverage - penalty
    )  # Return negative to minimize negative coverage (uncovered areas), penalize invalid points


# Optimization driver function using scipy.optimize.differential_evolution & .scipy.optimize.minimum
def visibility_optimized_points_3d(
    dem, valid_mask, n_points, method, obs_height, max_radius, num_rays, fixed_points
):
    valid_points = np.column_stack(np.where(valid_mask))

    dem_height, dem_width = dem.shape
    bounds = [(0, dem_height - 1), (0, dem_width - 1)] * n_points

    # Some plotting stuff - persistent figure and update in the window inplace (mpl interacive mode) to see the optimization in real-time
    plt.ion()  # interactive mode

    # Begin the differential evolution
    result = differential_evolution(
        objective,
        bounds,
        args=(
            dem,
            obs_height,
            max_radius,
            num_rays,
            n_points,
            valid_mask,
            fixed_points,
            method,
        ),
        init="halton",  # Several options avalible including: 'latinhypercube' (efficent) & 'sobol' (accurate)
        strategy="best1bin",  # Several options avalible. 'best1bin' is sufficent for most optimization
        maxiter=1000,  # Maximum number of iterations before ending
        popsize=8,  # Multiplier of population size - impacted by initilization 5 - 15 has workedd well
        tol=0.05,  # Tolerence threshold to stop iterating
        polish=True,  # Default true - polish with scipy.minimize
        workers=6,  # CPU cores to use
        updating="deferred",  # If "intermediate" - able to check for better solution within generation, "deferred" necessary if multiple worrkers
        disp=True,  # return feedback on state of optimizer as it is running (iteration and score)
        callback=callback,  # callback function (above) for live updates as the optimizer runs
    )

    plt.ioff()  # Turn off interactive mode when done

    optimized_points = result.x.reshape(-1, 2)
    observation_points = [(point[0], point[1]) for point in optimized_points]
    # observation_points = [(int(round(point[0])), int(round(point[1]))) for point in optimized_points]    # Rounded for now, may not want to do this

    observation_polygons = []
    fixed_polygons = []

    if method == "3d_poly":  # Use simplified (efficent) view area polygon approximaiton
        for obs_x, obs_y in observation_points:
            obs_visible_area = get_visible_area_3d_poly(
                dem, obs_x, obs_y, obs_height, max_radius, num_transects
            )
            observation_polygons.append(obs_visible_area)
        for fix_x, fix_y in fixed_points:
            fix_visible_area = get_visible_area_3d_poly(
                dem, fix_x, fix_y, obs_height, max_radius, num_transects
            )
            fixed_polygons.append(fix_visible_area)
    elif (
        method == "3d_poly_with_obstructions"
    ):  # Use enhanced (more accurate) view area polygon approximaiton
        for obs_x, obs_y in observation_points:
            obs_visible_area = get_visible_area_3d_with_obstructions(
                dem, obs_x, obs_y, obs_height, max_radius, num_transects
            )
            observation_polygons.append(obs_visible_area)
        for fix_x, fix_y in fixed_points:
            fix_visible_area = get_visible_area_3d_with_obstructions(
                dem, fix_x, fix_y, obs_height, max_radius, num_transects
            )
            fixed_polygons.append(fix_visible_area)
    else:
        raise ValueError(
            "Invalid method specified. Must be one of: '3d_poly','3d_poly_with_obstructions'"
        )

    return observation_points, fixed_points, observation_polygons, fixed_polygons


if __name__ == "__main__":
    dem_size = 100
    plot_generated_dem(*generate_dem(dem_size))

    dem, valid_mask, x, y = generate_dem(dem_size)

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

    # Fixed points that we will optimize around
    fixed_points = np.array(
        [[40, 30], [85, 70]]
    )  # locations of pre-existing points (crds in float or int)

    fig_op, ax_op = plt.subplots(
        1, 1, figsize=(8, 6)
    )  # figure and axis we will be updating

    n_points = 2  # Int: number of new observation points to place to increase coverage
    obs_height = 10  # Float or Int: Height of transmitter/ sensor - in map units, for ALL towers (including pre existing)
    obs_max_radius = 40  # Float or Int: Maximum transmission/view distance from tower
    n_rays = 90  # Int: Number of rays cast to generate polygon - complex terrain at distance will require more rays, defaut 90 (4 degrees between rays)
    method = "3d_poly"  # Two options: '3d_poly' (faster by about 8x), '3d_poly_with_obstructions' (more accurate, especially in complex terrain)

    obs_pts = visibility_optimized_points_3d(
        dem=dem,
        valid_mask=valid_mask,
        n_points=n_points,
        method=method,
        obs_height=obs_height,
        max_radius=obs_max_radius,
        num_rays=n_rays,
        fixed_points=fixed_points,
    )
    observation_points, fixed_points, observation_polygons, fixed_polygons = obs_pts
    print(observation_points)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_observation_points_with_polygons_3d(
        dem=dem,
        valid_mask=valid_mask,
        observation_points=observation_points,
        fixed_points=fixed_points,
        observation_polygons=observation_polygons,
        fixed_polygons=fixed_polygons,
        obs_height=obs_height,
        max_radius=obs_max_radius,
        num_rays=n_rays,
        ax=ax,
    )

    plt.show()

    # Create interactive widgets
    elev_slider = widgets.FloatSlider(
        min=0, max=90, step=1, value=30, description="Elevation"
    )
    azim_slider = widgets.FloatSlider(
        min=-180, max=180, step=1, value=45, description="Azimuth"
    )

    # Inject CSS for cell height
    display(
        HTML(
            """
    <style>
    .jp-OutputArea {
        max-height: 800px;
        overflow-y: auto;
    }
    </style>
    """
        )
    )

    # Display interactive plot
    interact(
        plot_dem_3d,
        dem=widgets.fixed(dem),
        valid_mask=widgets.fixed(valid_mask),
        observation_points=widgets.fixed(observation_points),
        observation_polygons=widgets.fixed(observation_polygons),
        fixed_points=widgets.fixed(fixed_points),
        fixed_polygons=widgets.fixed(fixed_polygons),
        obs_height=widgets.fixed(obs_height),
        elev=elev_slider,
        azim=azim_slider,
    )
