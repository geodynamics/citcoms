#!/usr/bin/env python2.7
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Simon Williams, John Cannon 
#
#                  ---------------------------------
#             (c) California Institute of Technology 2014
#             (c) The University of Sydnye 2014
#                        ALL RIGHTS RESERVED
#=====================================================================

# NOTE: this script requires python 2 to use pygplates 
import pygplates

import Core_Util_Python2
from Core_Util_Python2 import now 

# general python imports 
import datetime, os, pprint, re, subprocess, string, sys, traceback, math, copy

from scipy.io import netcdf_file as netcdf
from scipy.interpolate import RectBivariateSpline

#=====================================================================
#=====================================================================
def GetReconstructedMultipoint(reconstructed_feature_geometry):

    # Get the reconstructed geometry and the associated present day geometry.
    reconstructed_multipoint_geometry = reconstructed_feature_geometry.get_reconstructed_geometry()

    present_day_multipoint_geometry = pygplates.get_geometry_from_property_value(
            reconstructed_feature_geometry.get_property().get_value(),
            pygplates.MultiPointOnSphere)

    reconstructed_points = list(reconstructed_multipoint_geometry)
    present_day_points = list(present_day_multipoint_geometry)
    reconstructed_lat_points = []
    reconstructed_lon_points = []
    present_day_lat_lon_points = []

    # Iterate over the points in both multipoints (they should both have the same number of points).
    num_points = len(reconstructed_multipoint_geometry)
    #print now(), 'frame_change_pygplates: GetReconstructedMultipoint: num points in multipoint: %d' % num_points

    for point_index in range(0, num_points):
        # Index into the multipoint to get pygplates.PointOnSphere's.
        reconstructed_lat_lon_mp = pygplates.convert_point_on_sphere_to_lat_lon_point(reconstructed_points[point_index])
        reconstructed_lat_points.append(reconstructed_lat_lon_mp.get_latitude())
        reconstructed_lon_points.append(reconstructed_lat_lon_mp.get_longitude())
        present_day_lat_lon_mp = pygplates.convert_point_on_sphere_to_lat_lon_point(present_day_points[point_index])
        present_day_lat_lon_points.append((present_day_lat_lon_mp.get_latitude(), present_day_lat_lon_mp.get_longitude()))

    return (reconstructed_lon_points, reconstructed_lat_points,present_day_lat_lon_points)
#=====================================================================
#=====================================================================


#=====================================================================
# START of 'main'
#=====================================================================

# Get the GPlates model from the defaults file 
framework_d = Core_Util_Python2.parse_geodynamic_framework_defaults()

# verbose 
#Core_Util_Python2.tree_print(framework_d)

frame_change_input_rotation_filename = framework_d.get('frame_change_input_rotation_filename')
frame_change_input_multipoint_filename = framework_d.get('frame_change_input_multipoint_filename')

print now(), 'frame_change_pygplates: frame_change_input_rotation_filename =', frame_change_input_rotation_filename
print now(), 'frame_change_pygplates: frame_change_input_multipoint_filename =', frame_change_input_multipoint_filename


# read the rotation data 
rotation_model = pygplates.RotationModel(frame_change_input_rotation_filename)

# read the multipoint data 
print now(), 'frame_change_pygplates: Reading multipoint data...'
file_registry = pygplates.FeatureCollectionFileFormatRegistry()
multipoint_feature_collection = file_registry.read(frame_change_input_multipoint_filename)

print now(), 'frame_change_pygplates: Processing user input:'


# Get the recon time
reconstruction_time = int( sys.argv[1] )
print now(), 'frame_change_pygplates: reconstruction_time = ', reconstruction_time

# Get the input grid
input_grid_file = sys.argv[2]
print now(), 'frame_change_pygplates: input_grid_file = ', input_grid_file

# Get the grid -R 
input_grid_R = sys.argv[3]

input_grid = netcdf(input_grid_file,'r')

print now(), 'frame_change_pygplates: input_grid: variables', input_grid.variables


# set output filenames
output_xy_name = 'prefix-plate_frame.xy'

# check on filenames 
if input_grid_file.endswith('.nc') : 
    output_xy_name = input_grid_file.replace('.nc', '-plateframe.xy')
elif input_grid_file.endswith('.grd') :
    output_xy_name = input_grid_file.replace('.grd', '-plateframe.xy')

# write the output data file 
with open(output_xy_name, 'w') as output_xy_file:

    print now(), 'frame_change_pygplates: Create RectBivariateSpline of input grid'

    # Create an interpolation object for this grid
    f=RectBivariateSpline(
        input_grid.variables['lon'][:], 
        input_grid.variables['lat'][:], 
        input_grid.variables['z'][:].T 
    )

    # Reconstruct the multipoint features into a list of pygplates.ReconstructedFeatureGeometry's.
    reconstructed_feature_geometries = []

    print now(), 'frame_change_pygplates: pygplates.reconstruct multipoint_feature_collection'
    pygplates.reconstruct(multipoint_feature_collection, rotation_model, reconstructed_feature_geometries, reconstruction_time)

    print now(), 'frame_change_pygplates: num reconstructed multipoint geometries: %d' % len(reconstructed_feature_geometries)

    # loop over RFGs
    for reconstructed_feature_geometry in reconstructed_feature_geometries:

        # Get the recon positions 
        reconstructed_lon_points, reconstructed_lat_points, present_day_lat_lon_points = GetReconstructedMultipoint(reconstructed_feature_geometry)

        num_points = len(reconstructed_lon_points)
        # too verbose ? 
        # print now(), 'frame_change_pygplates: num points in multipoint: %d' % num_points

        # evaluate the current grid at the multipoint coordinates of the current feature
        gridZr = f.ev(reconstructed_lon_points, reconstructed_lat_points)

        # append the interpolated points as lon,lat,Z triples to an ascii .xy file
        for point_index in range(0, num_points):
            pdp = present_day_lat_lon_points[point_index]
            output_xy_file.write('%f %f %f\n' % (pdp[1], pdp[0], gridZr[point_index]))

output_grd_name = output_xy_name.replace('.xy', '.grd')
cmd = "nearneighbor %(output_xy_name)s -G%(output_grd_name)s -R%(input_grid_R)s -I0.5d -N1 -S0.75d -V" % vars()
print now(), 'frame_change_pygplates: cmd = ', cmd
os.system(cmd)

sys.exit(0)
