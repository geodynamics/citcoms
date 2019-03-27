#!/usr/bin/env python

import pygplates

input_rotation_filename = 'MODELS/RotationFiles/Global_EarthByte_TPW_CK95G94_2013.2.rot'
input_multipoint_filename = 'MODELS/StaticPolygons/lat_lon_velocity_domain_360_2013.2.shp'


rotation_model = pygplates.RotationModel(input_rotation_filename)

file_registry = pygplates.FeatureCollectionFileFormatRegistry()
print 'Reading multipoint data...'
multipoint_feature_collection = file_registry.read(input_multipoint_filename)

# get the recon time
reconstuction_time = sys.argv[1]

# read input grid
input_grid_file = sys.argv[2]
data = netcdf(InputGridFile,'r')

with open('output', 'w') as output_file:

    # Create an interpolation object for this grid
    f=RectBivariateSpline(data.variables['lon'][:],data.variables['lat'][:],data.variables['z'][:].T)

    # Reconstruct the multipoint features into a list of pygplates.ReconstructedFeatureGeometry's.
    reconstructed_feature_geometries = []
    pygplates.reconstruct(multipoint_feature_collection, rotation_model, reconstructed_feature_geometries, reconstruction_time)
    print 'num reconstructed multipoint geometries: %d' % len(reconstructed_feature_geometries)

    for reconstructed_feature_geometry in reconstructed_feature_geometries:

        reconstructed_lon_points, reconstructed_lat_points, present_day_lat_lon_points = GetReconstructedMultipoint(reconstructed_feature_geometry)
        num_points = len(reconstructed_lon_points)

        # evaluate the current grid at the multipoint coordinates of the current feature
        gridZr = f.ev(reconstructed_lon_points, reconstructed_lat_points)

        # append the interpolated points as lon,lat,Z triples to an ascii file
        for point_index in range(0, num_points):
            pdp = present_day_lat_lon_points[point_index]
            output_file.write('%f %f %f\n' % (pdp[1], pdp[0], gridZr[point_index]))



    
cmd = "nearneighbor output -G%sPlateFrameGrid%d.nc -Rd -I0.5d -N1 -S0.75d -V" % (output_file_stem,reconstruction_time)
os.system(cmd)


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
    # print 'num points in multipoint: %d' % num_points
    for point_index in range(0, num_points):
        # Index into the multipoint to get pygplates.PointOnSphere's.
        reconstructed_lat_lon_mp = pygplates.convert_point_on_sphere_to_lat_lon_point(reconstructed_points[point_index])
        reconstructed_lat_points.append(reconstructed_lat_lon_mp.get_latitude())
        reconstructed_lon_points.append(reconstructed_lat_lon_mp.get_longitude())
        present_day_lat_lon_mp = pygplates.convert_point_on_sphere_to_lat_lon_point(present_day_points[point_index])
        present_day_lat_lon_points.append((present_day_lat_lon_mp.get_latitude(), present_day_lat_lon_mp.get_longitude()))

    return (reconstructed_lon_points, reconstructed_lat_points,present_day_lat_lon_points)
#=====================================================================


