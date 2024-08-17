# Spatial_optimizer.py
from __future__ import annotations

# Syntax sugar stuff
# Handle some function loading stuff for the optimizer
from functools import partial
from typing import Optional

# Base imports
import geopandas as gpd  # v 0.14.4
import matplotlib.pyplot as plt  # Visulization - v 3.8.4
import numpy as np  # Array manipulation - v 1.26.4

# Visulization libs outside of mpl
from IPython.display import clear_output

# Scipy imports for spatial optimization
from scipy.ndimage import gaussian_filter  # Smoothing - v 1.13.0
from scipy.optimize import differential_evolution  # v 1.13.0

# Shapely imports for polygon handling and unions
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union

# Import all of our plotting functions from ploting_utils.py, these take up a LOT of lines, moved for readability
from plotting_utils import (
    plot_generated_surface,
    plot_observation_points_with_polygons_3d,
)


def generate_surface(map_size: int = 100):
    """
    Return a square test dataset (surface), valid areas, and coordinates in the form of numpy arrays.
    The resultant test data somewhat mimics real-world terrain.

    Keyword arguments:
    map_size -- size of the x AND y dimension of the test data
    """

    # Generate structured surface data coordinates, named xx & yy by convention
    xx, yy = np.meshgrid(np.linspace(-10, 10, map_size), np.linspace(-10, 10, map_size))

    # Create basin and range structure using combined functions and Gaussian filters
    generated_surface = (
        np.sin(xx) * np.cos(yy) * 20
        + np.sin(0.5 * xx) * np.cos(0.5 * yy) * 10
        + np.sin(0.25 * xx) * np.cos(0.25 * yy) * 5
    )

    # Apply gaussian filter to smooth out the surface
    generated_surface = gaussian_filter(generated_surface, sigma=2)

    # Add aggressive cliffs and valleys
    cliffs = (np.tanh(2 * (xx - 5)) - np.tanh(2 * (xx + 5))) * 20
    valleys = -(np.tanh(2 * (yy - 5)) - np.tanh(2 * (yy + 5))) * 20
    generated_surface += cliffs + valleys

    # Apply another gaussian filter for smooth transitions
    generated_surface = gaussian_filter(generated_surface, sigma=1)

    # Add random noise
    generated_surface += np.random.rand(map_size, map_size) * 5

    # Add lakes (negative values will be below sea level)
    generated_surface -= (
        np.exp(-0.05 * ((xx - 2) ** 2 + (yy - 2) ** 2)) * 30
        + np.exp(-0.05 * ((xx + 2) ** 2 + (yy + 2) ** 2)) * 30
    )

    # Normalize surface to be within 0 to 100 feet range
    generated_surface = (
        100
        * (generated_surface - np.min(generated_surface))
        / (np.max(generated_surface) - np.min(generated_surface))
    )

    # Prioritize center of mask using a gaussian dist as starting point
    valid_mask_base = np.exp(-0.1 * (xx**2 + yy**2))

    # Add blocks of random noise and apply gaussian filter to create blobs
    np.random.seed(42)  # For reproducibility
    noise = np.random.rand(map_size, map_size)
    smooth_noise = gaussian_filter(
        noise, sigma=10
    )  # Adjust sigma for larger/smaller blobs of valid or invalid areas
    generated_valid_mask = (
        valid_mask_base + smooth_noise
    ) > 0.5  # Combine and threshold to get a binary result

    return generated_surface, generated_valid_mask, xx, yy


def get_visible_area_3d_poly(
    surface: np.ndarray,
    obs_x: int | float,
    obs_y: int | float,
    obs_height: int | float,
    max_radius: int | float,
    num_transects: Optional[int] = 90,
) -> ShapelyPolygon:
    """
    Return a Shaply polygon based on a raytracing approach from an observation point.
    Polygon in based on num_transects and a max distance. line of sight is
    calculated by sweeping a 0 degrees elevation (i.e. laser level approach)

    Arguments:
    surface    -- surface surface, 2d array of elevation values
    obs_x      -- x coordinate in surface crds
    obs_y      -- y coordinate in surface crds
    obs_height -- additional "tower" height above the surface at (obs_x, obs_y)
    max_radius -- maximum distance a transect (ray) is cast, whereafter the max radius will be assumed

    Keyword Arguments:
    num_transects -- Int -- (default 90) number of instances to be checked, divided over 360 degrees
    """

    polygon_points = []
    surface_shape_0, surface_shape_1 = (
        surface.shape
    )  # Caching outside loop to avoid repeated lookups

    # Loop through our angles and distances (steps).
    # This nested loop isn't the most efficent, we should probally unroll it
    for angle in np.linspace(0, 2 * np.pi, num_transects, endpoint=False):
        dx = np.cos(angle)
        dy = np.sin(angle)
        obs_elevation = surface[
            int(np.round(obs_x)), int(np.round(obs_y))
        ]  # Convert to integers for approximaton

        max_distance = 0
        max_x, max_y = obs_x, obs_y

        # Check if angle/step has intersected terrain, at surface boundery, or at max_radius
        for step in range(1, max_radius + 1):
            x_flt = obs_x + step * dx
            y_flt = obs_y + step * dy
            x_int = int(round(x_flt))  # Ints for checking if we are at map boundery
            y_int = int(round(y_flt))
            if (
                x_int < 0
                or y_int < 0
                or x_int >= surface_shape_0
                or y_int >= surface_shape_1
            ):
                break

            distance_sq = (x_flt - obs_x) ** 2 + (y_flt - obs_y) ** 2
            target_elevation = surface[x_int, y_int]
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


def create_callback(method: str, kwargs):
    """
    Returns a callback function with access to the kwargs via closure
    """

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        """
        Plot the current best solution of the optimizer thus far. Callback is purely
        for progrss and visulization, not necessary for core functionality.

        Arguments:
        intermediate_result -- Supplied by differential_evolution
        """
        # Extract current best solution coordinates to a listkl:w

        current_solution = intermediate_result.x
        current_solution = [
            (int(current_solution[i]), int(current_solution[i + 1]))
            for i in range(0, len(current_solution), 2)  # Step through coordinate pairs
        ]

        observation_polygons = []
        fixed_polygons = []

        if (
            method == "3d_poly"
        ):  # Use simplified (efficent) view area polygon approximaton
            for obs_x, obs_y in current_solution:
                obs_visible_area = get_visible_area_3d_poly(dem, obs_x, obs_y, **kwargs)
                observation_polygons.append(obs_visible_area)
            for fix_x, fix_y in fixed_points:
                fix_visible_area = get_visible_area_3d_poly(
                    dem,
                    fix_x,
                    fix_y,
                    **kwargs,
                )
                fixed_polygons.append(fix_visible_area)
        else:
            raise ValueError(
                "Invalid method specified. Must be one of: '3d_poly','3d_poly_with_obstructions'"
            )

        # Clear previous plot so we see an "updating" plot
        clear_output(wait=True)

        # Plot the current solution
        plot_observation_points_with_polygons_3d(
            surface=dem,
            valid_mask=valid_mask,
            observation_points=current_solution,
            fixed_points=fixed_points,
            observation_polygons=observation_polygons,  # pass an empty list if you don't want to plot the polys when optimizing
            fixed_polygons=fixed_polygons,
            annotation=f"{current_solution}\ndifferential_evolution: f(x){intermediate_result.fun}",
            ax=ax_op,
        )

        plt.pause(0.1)  # Pause to allow the plot to update

    return callback


def objective(
    params: np.ndarray,
    surface: np.ndarray,
    valid_mask: np.ndarray,
    fixed_points: list[list[int | float]],
    method: Optional[str] = "3d_poly",
    **kwargs,
) -> float:
    """
    Return the score of the current solution. Used as the fitness function for
    differential_evolution
    - Prioriries
    - 1) Do not place on invalid areas (harsh penelty will prevent this)
    - 2) Map coverage - we desire high coverage for the given n points (what we are optimizing for)

    Arguments:
    params        -- A 1D array containing the x and y coordinates of
                     the points being optimized as determined by differential_evolution,
                     THIS IS SUPPLIED BY differential_evolution (initially derived ferom bounds)
    surface       -- 2d float array of surface elevation
    valid_mask    -- 2d boolean array of the valid_mask cost surface
    fixed_points  -- Coordinate pairs of fixed observation points not allowed to change
    **kwargs      -- Additional arguments passed to through the fitness function (objective) to get_visible_area_3d_poly()

    Keyword Arguments:
    method        -- User defined function id for calculating the visible area
    """
    points = params.reshape(
        -1, 2
    )  # unflatten the points supplied by differential_evolution
    visible_area_polys = []
    penalty = 0  # Initialize penalty, start at zero

    # Include fixed points in the points to be evaluated to get accurate coverage
    all_points = np.vstack((points, fixed_points))

    for i, (obs_x, obs_y) in enumerate(all_points):
        # Ensure indices are within bounds and convert to integer (for speed)
        obs_x = int(np.clip(obs_x, 0, surface.shape[1] - 1))
        obs_y = int(np.clip(obs_y, 0, surface.shape[0] - 1))

        # Only check validity for observation points
        if i < len(points):
            # Check if the point is in a valid area, if this were a continious cost surface this check would need to change
            if valid_mask[obs_x, obs_y] == 0:
                penalty += 1000  # Add large penalty for invalid point, this can be modified for a continious cost surface (not binary/thresholded)

        if method == "3d_poly":
            visible_area = get_visible_area_3d_poly(
                surface=surface,
                obs_x=obs_x,
                obs_y=obs_y,
                **kwargs,
            )
        else:
            raise ValueError(
                "Invalid method specified, methods trimmed for example code. Must be one of: '3d_poly'"
            )

        # Handle potential invalid shapely geometries from our visibility functions, buffer helps resolve
        if not visible_area.is_valid:
            visible_area = visible_area.buffer(0)

        visible_area_polys.append(visible_area)

    if visible_area_polys:
        # Calculate total visible area (map coverage) if visible_area_polys is not empty
        map_coverage = (gpd.GeoSeries(unary_union(visible_area_polys)).area).iloc[
            0
        ] / np.prod(surface.shape)
    else:
        map_coverage = 0  # return 0 if there is no valid visible_area_polys (this is an edge case, not likely we enter here)

    return -(
        map_coverage - penalty
    )  # Return negative to minimize negative coverage (uncovered areas), penalize invalid points


def visibility_optimized_points_3d(
    surface: np.ndarray,
    valid_mask: np.ndarray,
    n_points: int,
    method: str,
    fixed_points: list[list[int | float]],
    **kwargs,
) -> tuple[
    list[list[int | float]],
    list[list[int | float]],
    list[ShapelyPolygon],
    list[ShapelyPolygon],
]:
    """
    Return:
    observation_points   -- Found optimized points coordinates, float or int
    fixed_points         -- Fixed point coordinates, as we passed in, float or int
    observation_polygons -- Visibility coverage polygon for optimized points
    fixed_polygons       -- Same as observation_polygons, but for fixed points

    Arguments:
    n_points             -- Number of points to place/ optimize
    surface              -- 2d float array of surface elevation
    valid_mask           -- 2d boolean array of the valid_mask cost surface
    method               -- User defined function id for calculating the visible area (e.g. "3d_poly")
    fixed_points         -- Coordinate pairs of fixed observation points not allowed to change
    **kwargs             -- Additional arguments passed to through the fitness function (objective) to get_visible_area_3d_poly()
    """

    surface_height, surface_width = (
        surface.shape
    )  # use the surface size as our maximum bounds

    # Create n_points number of bounds coordinates, one set for each n_point
    bounds = [
        (0, surface_height - 1),
        (0, surface_width - 1),
    ] * n_points  # This could also be an instance of the scipy.optimize.Bounds class

    # Some plotting stuff - persistent figure and update in the window inplace (mpl interacive mode) to see the optimization in real-time
    plt.ion()  # interactive mode to make updating the plot in a singular window possible

    # Create a partial function to include fixed arguments and keyword args
    # that differential evolution can use, this allows us to pass **kwargs
    # in differential_evolution(), which has a strict args=(...) pattern,
    objective_with_args = partial(
        objective,
        surface=surface,
        valid_mask=valid_mask,
        fixed_points=fixed_points,
        method=method,
        **kwargs,
    )

    # Create a callback function w/ access to the kwargs since by default the callback is VERY limited
    callback_with_kwargs = create_callback(method, kwargs)

    # Begin the differential evolution
    result = differential_evolution(
        objective_with_args,
        bounds,
        init="halton",  # Several options avalible including: 'latinhypercube' (efficent) & 'sobol' (accurate)
        strategy="best1bin",  # Several options avalible. 'best1bin' is sufficent for most optimization
        maxiter=1000,  # Maximum number of iterations before ending
        popsize=10,  # Multiplier of population size - impacted by initilization 5 - 15 has worked well
        tol=0.05,  # Tolerence threshold to stop iterating
        polish=True,  # Default true - polish with scipy.minimize
        workers=6,  # CPU cores to use
        updating="deferred",  # If "intermediate" - able to check for better solution within generation, "deferred" necessary if multiple worrkers
        disp=True,  # Return feedback on state of optimizer on terninal as it is running (iteration and score)
        callback=callback_with_kwargs,  # (optional) callback function (above) for live updates as the optimizer runs
    )

    plt.ioff()  # Turn off interactive mode when done

    optimized_points = result.x.reshape(-1, 2)  # reshape back to coordinat pairs
    observation_points = [(point[0], point[1]) for point in optimized_points]

    observation_polygons = []
    fixed_polygons = []

    if method == "3d_poly":  # Use simplified (efficent) view area polygon approximaiton
        for obs_x, obs_y in observation_points:
            obs_visible_area = get_visible_area_3d_poly(surface, obs_x, obs_y, **kwargs)
            observation_polygons.append(obs_visible_area)
        for fix_x, fix_y in fixed_points:
            fix_visible_area = get_visible_area_3d_poly(
                surface,
                fix_x,
                fix_y,
                **kwargs,
            )
            fixed_polygons.append(fix_visible_area)
    else:
        raise ValueError(
            "Invalid method specified. Must be one of: '3d_poly','3d_poly_with_obstructions'"
        )

    return observation_points, fixed_points, observation_polygons, fixed_polygons


if __name__ == "__main__":
    dem_size = 100
    dem, valid_mask, x, y = generate_surface(dem_size)
    plot_generated_surface(dem, valid_mask)

    # Fixed points (y, x) that we will optimize other points, fixed points do not move
    fixed_points = np.array(
        [[5, 17], [85, 80]]
    )  # locations of pre-existing points (crds in float or int)

    fig_op, ax_op = plt.subplots(
        1, 1, figsize=(8, 6)
    )  # figure and axis we will be updating to visulize the optimization in progress

    n_points = 2  # Int: number of new observation points to place to increase coverage, these will be moved as we optimize
    polygon_method = "3d_poly"  # Two options: '3d_poly' (faster by about 8x) and  <removed for brevity>'3d_poly_with_obstructions' (more accurate, especially in complex terrain)
    pt_height = 5  # Float or Int: Height of transmitter/ sensor - in map units, for ALL towers (including pre existing)
    pt_max_radius = 30  # Float or Int: Maximum transmission/view distance from tower
    pt_n_transects = 90  # Int: Number of rays (transects) cast to generate polygon - complex terrain at distance will require more rays, defaut 90 (4 degrees between rays). This can roughly be considered analogous to camera resolution

    obs_pts = visibility_optimized_points_3d(
        surface=dem,
        valid_mask=valid_mask,
        n_points=n_points,
        fixed_points=fixed_points,
        method=polygon_method,
        # Additional kwargs to pass to the spatial evaluation.
        # These should ultimatly be part of a class with properties if any more complex than ~3 parameters
        obs_height=pt_height,
        max_radius=pt_max_radius,
        num_transects=pt_n_transects,
    )
    observation_points, fixed_points, observation_polygons, fixed_polygons = obs_pts
    print(observation_points)

    # display final plot with optimized points and coverage polygons
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    plot_observation_points_with_polygons_3d(
        surface=dem,
        valid_mask=valid_mask,
        observation_points=observation_points,
        fixed_points=fixed_points,
        observation_polygons=observation_polygons,
        fixed_polygons=fixed_polygons,
        ax=ax,
    )

    plt.show()
# EOF
