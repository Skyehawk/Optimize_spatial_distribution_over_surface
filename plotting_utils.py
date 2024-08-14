# plot_functions.py

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go


def plot_generated_dem(dem, valid_mask):
    """
    Return Matplotlib figure of the input arrays using imshow()

    Args:
    dem        -- Numpy ndarray - representing the terrain surface (2 dimesional)
    valid_mask -- Numpy ndarray - representing the valid areas mask (2 dimentional)

    """
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    c1 = axs[0].imshow(
        dem, cmap="terrain", origin="lower"
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
    dem,
    valid_mask,
    observation_points,
    fixed_points,
    observation_polygons,
    fixed_polygons,
    obs_height=15,
    max_radius=75,
    num_transects=90,
    annotation=None,
    ax=None,
):
    if ax is not None and not isinstance(ax, plt.Axes):
        raise TypeError(
            "Expected None or a matplotlib Axes instance, but got type '{}'".format(
                type(ax).__name__
            )
        )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.get_figure()

    ax.cla()

    # Create grid coordinates for pcolormesh
    x, y = np.meshgrid(
        np.linspace(0, dem.shape[1], dem.shape[1] + 1),
        np.linspace(0, dem.shape[0], dem.shape[0] + 1),
    )

    # Plot DEM as pcolormesh
    ax.pcolormesh(x, y, dem, cmap="terrain", shading="auto")
    ax.set_title("Digital Elevation Model (DEM)")
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
        # Add annotation at the bottom center
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
    # ax.legend()
    # ax.show()
    plt.pause(1e-10)


# Function to create the 3D plot using Plotly


def plot_dem_3d(
    dem,
    valid_mask,
    observation_points,
    fixed_points,
    observation_polygons,
    fixed_polygons,
    obs_height,
    elev=30,
    azim=45,
):
    x = np.arange(dem.shape[1])
    y = np.arange(dem.shape[0])
    x, y = np.meshgrid(x, y)
    z = dem

    # Create surface plot for DEM
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y, colorscale="tropic", opacity=0.8)])

    # Overlay valid areas
    fig.add_trace(go.Surface(z=z, x=x, y=y, surfacecolor=valid_mask, opacity=0.3))

    # Plot observation points
    obs_x = [p[1] for p in observation_points]
    obs_y = [p[0] for p in observation_points]
    obs_z = [dem[int(p[0]), int(p[1])] + obs_height for p in observation_points]
    fig.add_trace(
        go.Scatter3d(
            x=obs_x,
            y=obs_y,
            z=obs_z,
            mode="markers",
            marker=dict(size=5, color="red"),
            name="Observation Points",
        )
    )

    # Plot fixed points
    fixed_x = [p[1] for p in fixed_points]
    fixed_y = [p[0] for p in fixed_points]
    fixed_z = [dem[int(p[0]), int(p[1])] + obs_height for p in fixed_points]
    fig.add_trace(
        go.Scatter3d(
            x=fixed_x,
            y=fixed_y,
            z=fixed_z,
            mode="markers",
            marker=dict(size=5, color="blue"),
            name="Fixed Points",
        )
    )

    # Add lines from observation points to terrain surface
    for p in observation_points:
        obs_x, obs_y = p[1], p[0]
        obs_z_surface = dem[int(p[0]), int(p[1])]
        obs_z_point = obs_z_surface + obs_height
        fig.add_trace(
            go.Scatter3d(
                x=[obs_x, obs_x],
                y=[obs_y, obs_y],
                z=[obs_z_point, obs_z_surface],
                mode="lines",
                line=dict(color="red", width=2),
                showlegend=False,
            )
        )

    # Add lines from fixed points to terrain surface
    for p in fixed_points:
        fixed_x, fixed_y = p[1], p[0]
        fixed_z_surface = dem[int(p[0]), int(p[1])]
        fixed_z_point = fixed_z_surface + obs_height
        fig.add_trace(
            go.Scatter3d(
                x=[fixed_x, fixed_x],
                y=[fixed_y, fixed_y],
                z=[fixed_z_point, fixed_z_surface],
                mode="lines",
                line=dict(color="blue", width=2),
                showlegend=False,
            )
        )

    # Add polygons at the respective height of each observation point
    for idx, polygon in enumerate(observation_polygons):
        polygon_points = np.array(polygon.exterior.coords)
        polygon_points = np.column_stack(
            (
                polygon_points[:, 1],
                polygon_points[:, 0],
                np.full_like(polygon_points[:, 0], obs_z[idx]),
            )
        )
        fig.add_trace(
            go.Mesh3d(
                x=polygon_points[:, 0],
                y=polygon_points[:, 1],
                z=polygon_points[:, 2],
                opacity=0.15,
                color="salmon",
                name=f"Polygon {idx+1}",
            )
        )

    # Add polygons at the respective height of each fixed point
    for idx, polygon in enumerate(fixed_polygons):
        polygon_points = np.array(polygon.exterior.coords)
        polygon_points = np.column_stack(
            (
                polygon_points[:, 1],
                polygon_points[:, 0],
                np.full_like(polygon_points[:, 0], fixed_z[idx]),
            )
        )
        fig.add_trace(
            go.Mesh3d(
                x=polygon_points[:, 0],
                y=polygon_points[:, 1],
                z=polygon_points[:, 2],
                opacity=0.15,
                color="dodgerblue",
                name=f"Fixed Polygon {idx+1}",
            )
        )

    # Set layout
    fig.update_layout(
        title="3D View of DEM, Valid Areas, & Observation Points",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Elevation",
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.5),
            camera=dict(
                eye=dict(
                    x=np.sin(np.deg2rad(azim)) * np.cos(np.deg2rad(elev)),
                    y=np.sin(np.deg2rad(azim)) * np.sin(np.deg2rad(elev)),
                    z=np.cos(np.deg2rad(elev)),
                )
            ),
            xaxis=dict(range=[0, dem.shape[1]]),
            yaxis=dict(range=[0, dem.shape[0]]),
            zaxis=dict(range=[dem.min(), max(dem.max(), max(obs_z))]),
        ),
        width=1000,
        height=800,
    )

    # Adjust colorbar position and size
    fig.update_layout(coloraxis_colorbar=dict(yanchor="top", y=0.9, len=0.4))

    # Show plot
    fig.show()
