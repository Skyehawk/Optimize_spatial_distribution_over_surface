import numpy as np
cimport numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

def get_visible_area_3d_with_obstructions(np.ndarray[double, ndim=2] dem, int obs_x, int obs_y, double obs_height, int max_radius, int num_transects=90):
    cdef int dem_size_x = dem.shape[0]
    cdef int dem_size_y = dem.shape[1]
    cdef list polygon_points = []
    cdef list obstructed_points = []
    
    cdef double[:] angles = np.linspace(0, 2 * np.pi, num_transects, endpoint=False)
    cdef double[:] cos_angles = np.cos(angles)
    cdef double[:] sin_angles = np.sin(angles)
    cdef double obs_elevation = dem[int(np.round(obs_x)), int(np.round(obs_y))]
    
    cdef double dx, dy, x, y
    cdef int step, x_int, y_int, i, j, px, py
    cdef bint obstructed
    cdef double back_x, back_y, back_elevation, back_distance, line_elevation_back, max_x, max_y
    
    for i in range(num_transects):
        dx = cos_angles[i]
        dy = sin_angles[i]
        max_x, max_y = obs_x, obs_y
        transect_points = []
        
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
        
        for point in transect_points:
            px, py = point
            obstructed = False
            
            for step_back in range(1, len(transect_points) + 1):
                back_x = obs_x + step_back * (px - obs_x) / max_radius
                back_y = obs_y + step_back * (py - obs_y) / max_radius
                back_x_int = int(round(back_x))
                back_y_int = int(round(back_y))
                
                if back_x_int < 0 or back_y_int < 0 or back_x_int >= dem_size_x or back_y_int >= dem_size_y:
                    continue
                
                back_elevation = dem[back_x_int, back_y_int]
                back_distance = np.sqrt((back_x - obs_x)**2 + (back_y - obs_y)**2)
                line_elevation_back = (obs_elevation + obs_height) - ((obs_elevation + obs_height) - back_elevation) * (back_distance / max_radius)
                
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

