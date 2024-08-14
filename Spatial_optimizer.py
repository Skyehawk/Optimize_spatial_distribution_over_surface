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

# Import all of our plotting functions from ploting_utils.py, these take up a LOT of lines, moved for readability
from plotting_utils import (
    plot_dem_3d,
    plot_generated_dem,
    plot_observation_points_with_polygons_3d,
)


def generate_dem(map_size=100):
    """
    Return a square test dataset (dem), valid areas, and coordinates in the form of numpy arrays.
    The resultant test data somewhat mimics real-world terrain.

    Keyword arguments:
    map_size -- Int - size of the x and y dimension of the test data
    """

    # Generate structured DEM data coordinates
    xx, yy = np.meshgrid(np.linspace(-10, 10, map_size), np.linspace(-10, 10, map_size))

    # Create basin and range structure using combined functions and Gaussian filters
    generated_dem = (
        np.sin(xx) * np.cos(yy) * 20
        + np.sin(0.5 * xx) * np.cos(0.5 * yy) * 10
        + np.sin(0.25 * xx) * np.cos(0.25 * yy) * 5
    )

    # Apply Gaussian filter to smooth out the DEM
    generated_dem = gaussian_filter(generated_dem, sigma=2)

    # Add aggressive cliffs and valleys
    cliffs = (np.tanh(2 * (xx - 5)) - np.tanh(2 * (xx + 5))) * 20
    valleys = -(np.tanh(2 * (yy - 5)) - np.tanh(2 * (yy + 5))) * 20
    generated_dem += cliffs + valleys

    # Apply another Gaussian filter for smooth transitions
    generated_dem = gaussian_filter(generated_dem, sigma=1)

    # Add random noise
    generated_dem += np.random.rand(map_size, map_size) * 5

    # Add lakes (negative values will be below sea level)
    generated_dem -= (
        np.exp(-0.05 * ((xx - 2) ** 2 + (yy - 2) ** 2)) * 30
        + np.exp(-0.05 * ((xx + 2) ** 2 + (yy + 2) ** 2)) * 30
    )

    # Normalize DEM to be within 0 to 100 feet range
    generated_dem = (
        100
        * (generated_dem - np.min(generated_dem))
        / (np.max(generated_dem) - np.min(generated_dem))
    )

    # Prioritize center of mask using a Gaussian dist as starting point
    valid_mask_base = np.exp(-0.1 * (xx**2 + yy**2))

    # Add blocks of random noise and apply Gaussian filter to create blobs
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(map_size, map_size)
    smooth_noise = gaussian_filter(
        noise, sigma=10
    )  # Adjust sigma for larger/smaller blobs of valid or invalid areas
    generated_valid_mask = (
        valid_mask_base + smooth_noise
    ) > 0.5  # Combine and threshold to get a binary result

    return generated_dem, generated_valid_mask, xx, yy


def get_visible_area_3d_poly(
    dem, obs_x, obs_y, obs_height, max_radius, num_transects=90
):
    """
    Return a Shaply polygon based on a raytracing approach.
    Polygon in based on num_transects and a max distance. line of sight is
    calculated by sweeping a 0 degrees elevation (i.e. laser level approach)

    Arguments:
    dem        -- Numpy ndarray - dem surface, 2d array of elevation values
    obs_x      -- Float - x coordinate in dem crds
    obs_y      -- Float - y coordinate in dem crds
    obs_height -- Float - additional "tower" height above the surface at (obs_x, obs_y)
    max_radius -- Float - maximum distance a ray is cast, whereafter the max radius will be assumed

    Keyword Arguments:
    num_transects -- Int -- default=90, number of instances to be checked, divided over 360 degrees
    """

    polygon_points = []
    dem_shape_0, dem_shape_1 = dem.shape

    # Loop through our angles and distances (steps).
    # This nested loop isn't the most efficent, we should probally unroll it
    for angle in np.linspace(0, 2 * np.pi, num_transects, endpoint=False):
        dx = np.cos(angle)
        dy = np.sin(angle)
        obs_elevation = dem[
            int(np.round(obs_x)), int(np.round(obs_y))
        ]  # Convert to integers

        max_distance = 0
        max_x, max_y = obs_x, obs_y

        # Check if angle/step has intersected terrain, at dem boundery, or at max_radius
        for step in range(1, max_radius + 1):
            x_flt = obs_x + step * dx
            y_flt = obs_y + step * dy
            x_int = int(round(x_flt))  # Ints for checking if we are at map boundery
            y_int = int(round(y_flt))
            if x_int < 0 or y_int < 0 or x_int >= dem_shape_0 or y_int >= dem_shape_1:
                break

            distance_sq = (x_flt - obs_x) ** 2 + (y_flt - obs_y) ** 2
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


# Callback function for visulizing the results as the optimization progresses
def callback(intermediate_result):
    # Extract current best solution
    current_solution = intermediate_result.x
    current_solution = [
        (int(current_solution[i]), int(current_solution[i + 1]))
        for i in range(0, len(current_solution), 2)
    ]

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
        num_transects=n_transects,
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
    num_transects,
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
                penalty += 1000  # Add large penalty for invalid point

        if method == "3d_poly":
            visible_area = get_visible_area_3d_poly(
                dem=dem,
                obs_x=obs_x,
                obs_y=obs_y,
                obs_height=obs_height,
                max_radius=max_radius,
                num_transects=num_transects,
            )
        else:
            raise ValueError(
                "Invalid method specified, methods trimmed for example code. Must be one of: '3d_poly'"
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
    dem,
    valid_mask,
    n_points,
    method,
    obs_height,
    max_radius,
    num_transects,
    fixed_points,
):

    dem_height, dem_width = dem.shape
    bounds = [(0, dem_height - 1), (0, dem_width - 1)] * n_points

    # Some plotting stuff - persistent figure and update in the window inplace (mpl interacive mode) to see the optimization in real-time
    plt.ion()  # interactive mode to make updating the plot in a singular window possible

    # Begin the differential evolution
    result = differential_evolution(
        objective,
        bounds,
        args=(
            dem,
            obs_height,
            max_radius,
            num_transects,
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
    else:
        raise ValueError(
            "Invalid method specified. Must be one of: '3d_poly','3d_poly_with_obstructions'"
        )

    return observation_points, fixed_points, observation_polygons, fixed_polygons


if __name__ == "__main__":
    dem_size = 100
    dem, valid_mask, x, y = generate_dem(dem_size)
    plot_generated_dem(dem, valid_mask)

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
    n_transects = 90  # Int: Number of rays (transects) cast to generate polygon - complex terrain at distance will require more rays, defaut 90 (4 degrees between rays)
    method = "3d_poly"  # Two options: '3d_poly' (faster by about 8x), '3d_poly_with_obstructions' (more accurate, especially in complex terrain)

    obs_pts = visibility_optimized_points_3d(
        dem=dem,
        valid_mask=valid_mask,
        n_points=n_points,
        method=method,
        obs_height=obs_height,
        max_radius=obs_max_radius,
        num_transects=n_transects,
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
        num_transects=n_transects,
        ax=ax,
    )

    plt.show()

    # Create interactive widgets, This will launch a browser, uncomment to allow this
    '''
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
    '''
