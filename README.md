# Optimize_spatial_distribution_over_surface


Implementation of spatial optimization for n points over a surfce with surface interference (i.e. a hill blocking line of sight with a suite of transmitters)

* Goal is to cover surface (DEM) with "visibility" to observation points
* Points can be offset from surface, have a maximum view range, etc.
* View coverage is 3-dimentional (respects blockages such as a hill, or hidden areas such as a canyon)
* Placement can respect a cost surface - currently implemented as a binary (valid/invalid) surface


