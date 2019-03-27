#!/usr/bin/python
#
#<LicenseText>
#
#=====================================================================
#                              Author: Mark Turner
#          (c) California Institute of Technology 2014
#
#               Free for non-commercial academic use ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#=====================================================================
import sys, string, os

#
# Change these default values as needed 
#

feature_type='UnclassifiedFeature'
rotation_id = '0'
begin_age = '100.0'
end_age = '0.0'
geom_type='line'
coordinate_order=''

feature_name = 'sample feature' # this is a place holder
# feature names will be parsed from data on the '>' line, if available 

#
# GPML boiler plate ; DO NOT CHANGE 
#

gpml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<gpml:FeatureCollection xmlns:gpml="http://www.gplates.org/gplates" xmlns:gml="http://www.opengis.net/gml" xmlns:xsi="http://www.w3.org/XMLSchema-instance" gpml:version="1.6" xsi:schemaLocation="http://www.gplates.org/gplates ../xsd/gpml.xsd http://www.opengis.net/gml ../../../gml/current/base">
'''

# this string is fixed ; do not change
gpml_footer='''
</gpml:FeatureCollection>
'''


#
# Geometry types
#
gpml_position_as_point = '''
            <gpml:position>
                <gml:Point>
                    <gml:pos>%(pos_list)s</gml:pos>
                </gml:Point>
            </gpml:position>
'''


gpml_center_line_of_as_curve='''
            <gpml:centerLineOf>
                <gpml:ConstantValue>
                    <gpml:value>
                        <gml:OrientableCurve>
                            <gml:baseCurve>
                                <gml:LineString>
                                    <gml:posList gml:dimension="2">%(pos_list)s</gml:posList>
                                </gml:LineString>
                            </gml:baseCurve>
                        </gml:OrientableCurve>
                    </gpml:value>
                    <gpml:valueType xmlns:gml="http://www.opengis.net/gml">gml:OrientableCurve</gpml:valueType>
                </gpml:ConstantValue>
            </gpml:centerLineOf>
'''

gpml_outline_of_as_polygon = '''
            <gpml:outlineOf>
                <gpml:ConstantValue>
                    <gpml:value>
                        <gml:Polygon>
                            <gml:exterior>
                                <gml:LinearRing>
                                    <gml:posList gml:dimension="2">%(pos_list)s</gml:posList>
                                </gml:LinearRing>
                            </gml:exterior>
                        </gml:Polygon>
                    </gpml:value>
                    <gpml:valueType xmlns:gml="http://www.opengis.net/gml">gml:Polygon</gpml:valueType>
                </gpml:ConstantValue>
            </gpml:outlineOf>
'''


# this list of strings will be filled during processing
gpml_features = []

# =========================================================================
# =========================================================================
def convert_feature_dict_to_gpml( feature ):
    '''given feature dictionary, create gpml version, and add to the global list'''

    global gpml_features 
    global feature_type, rotation_id, begin_age, end_age, geom_type
    global feature_name
    global gpml_center_line_of_as_curve
    global gpml_outline_of_as_polygon

    # create local copies of the globals to use vars() syntax
    type = feature_type
    rid = rotation_id
    begin = begin_age
    end = end_age

    # set the geom type 
    if (geom_type == 'line') :
        geom = gpml_center_line_of_as_curve # default
    if (geom_type == 'polygon') :
        geom = gpml_outline_of_as_polygon

    # correct for points 
    if len( feature['points']) == 1:
        geom = gpml_position_as_point

    # set the name
    if feature.has_key('name'):
        name = feature['name']
    else: 
        name = feature_name

    # get the list of coordinates as a string 
    pos_list = feature['pos_list']

    part_1 = '''
    <gml:featureMember>
        <gpml:%(type)s>
            <gpml:reconstructionPlateId>
                <gpml:ConstantValue>
                    <gpml:value>%(rid)s</gpml:value>
                    <gpml:valueType xmlns:gpml="http://www.gplates.org/gplates">gpml:plateId</gpml:valueType>
                </gpml:ConstantValue>
            </gpml:reconstructionPlateId>
''' % vars()

    part_2 = geom % vars()

    part_3 = '''
            <gml:validTime>
                <gml:TimePeriod>
                    <gml:begin>
                        <gml:TimeInstant>
                            <gml:timePosition>%(begin)s</gml:timePosition>
                        </gml:TimeInstant>
                    </gml:begin>
                    <gml:end>
                        <gml:TimeInstant>
                            <gml:timePosition>%(end)s</gml:timePosition>
                        </gml:TimeInstant>
                    </gml:end>
                </gml:TimePeriod>
            </gml:validTime>
            <gml:name>%(name)s</gml:name>
        </gpml:%(type)s>
    </gml:featureMember>
''' % vars()

    gpml = part_1 + part_2 + part_3

    gpml_features.append( gpml )
# =========================================================================

# =========================================================================
def read_dot_xy( file_name ):
    ''' read a .xy file '''

    # open the file and read all the lines
    lines = []
    print 'open:', file_name
    file = open( file_name )
    try:
        lines = [ line.rstrip() for line in file if line[0] != '#' ]
    finally:
        file.close()

    # working vars to hold feature data
    list_of_features = [] # list of feature dictionaries 
    feature = {} # single dictionary 
    list_of_points = [] # list of points per feature; list of tuples


    # Some xy files will not have any header lines,
    # so assume a single feature per file will be created

    # start up a new feature; clear old data
    feature = {} 
    list_of_points = []
    feature['pos_list'] = '' # empty string

    # set the default feature name to be the input file name
    feature['name'] = file_name # set the name in the dictionary
    print "feature name =", feature['name']

    for l in lines:

        if l.startswith('>'):

            if len(list_of_points) > 0:
                # close out the previous feature 
                feature['points'] = list_of_points # set the point data 
                # make a deep copy with dict() on the list
                list_of_features.append( dict(feature) )

            # start up a new feature; clear old data
            feature = {} 
            list_of_points = []
            feature['pos_list'] = '' # empty string

            # start up new feature
            name = l
            name = name.replace('>','') # remove > 
            name = name.rstrip() # clean off any white space
            name = name.lstrip() # clean off any white space
            name = name.replace('&', '_'); # convert any '&' to '_'
            feature['name'] = name # set the name in the dictionary
            print "feature name =", name
            continue # to next line


        # regular lat lon data line, split and save
        s = l.split()
        lon = s[0]
        lat = s[1]

        # check for toggle on coordinates
        if (coordinate_order == '-:') :
            lat = s[0]
            lon = s[1]

        # update lists of points
        list_of_points.append( (lon, lat) )
        feature['pos_list'] += '%(lat)s %(lon)s ' % vars()

    # end of loop over lines

    # close out the last feature , 
    # in case the file did not have a final '>' to trigger the above copy
    if len(list_of_points) > 0:
        # set the point data 
        feature['points'] = list_of_points
        # make a deep copy with dict() on the list
        list_of_features.append( dict(feature) )

    return list_of_features

# =========================================================================
# =========================================================================
def write_features_as_gpml( list_of_features, out_file_name ):
    '''write out the global list points, as .gmpl'''

    global gpml_features
    gpml_features = []

    print 'write:', out_file_name
    file = open( out_file_name, 'w')
    file.write('%s' % gpml_header)

    for feature in list_of_features:
        #print "feature =", feature
        convert_feature_dict_to_gpml( feature )

    file.write('%s' % ''.join(gpml_features) )

    file.write('%s' % gpml_footer)
# =========================================================================

# =========================================================================
def usage(): 
    ''' simple script to convert roation data''' 
    print 
    print 'usage:'
    print sys.argv[0], "[-line] [-polygon] [-:] [-type] [-rid] -[begin] -[end] input_files "
    print 
    print "This script creates gpml features from raw coordinate data,"
    print "(typically in .xy or .dat formats) and used command line options,"
    print "and header data to set feature properties."
    print 
    print "The script reads one or more text files with coordinate data "
    print "in 'lon lat' columns (or 'lat lon' if the -: flag is is used; see below.),"
    print "and converts the data to a either line or polygon geometry."
    print 
    print "The output file names are the same as the input names,"
    print "with the '.gpml' sting appended."
    print 
    print "If GMT style '>' header lines are found, the script assumes"
    print "the input file contains a single feature."
    print  
    print "Multiple features may exist in a single input file,"
    print "as long as they are separated by '>' header lines."
    print 
    print "The name property of the feature will be read from header line,"
    print "ommiting the '>' character, and removing any whitespace found"
    print "between the '>' and text, and removing any trailing whitespace."
    print 
    print "optional arugments that apply to all features created:"
    print 
    print "Use -: to toggle between (longitude,latitude) and (latitude,longitude) input."
    print "[Default  is  (longitude,latitude)]."
    print 
    print "Use -line to create line features (default)"
    print 
    print "Use -polygon to create polygon features"
    print " NOTE: the raw coordinate data must have identical frist and last points "
    print " to form valid polygon geometry."
    print 
    print "Use -type to set a gpml feature type (default is UnclassifiedFeature)"
    print 
    print "Use -rid to set a rotation id (default is 0)"
    print 
    print "Use -begin to set the begin_age (default is 100)"
    print 
    print "Use -end to set the end_age (default is 0)"
    print 
    sys.exit()
# =========================================================================
# =========================================================================
if (__name__=='__main__'):

    skip_next_arg = False

    if len(sys.argv) < 2: 
        usage()
    else:
        list = sys.argv[1:] # remove the name of the script from the args
        for i,arg in enumerate( list ) :

            if skip_next_arg: 
                skip_next_arg = False
                continue; # to next arg

            if arg == '-line':
                geom_type = arg.replace('-','')
                continue # to next arg

            if arg == '-polygon':
                geom_type = arg.replace('-','')
                continue # to next arg

            if arg == '-:':
                coordinate_order = arg
                continue # to next arg

            if arg == '-type':
                feature_type = list[i+1]
                skip_next_arg = True
                continue # to next arg

            if arg == '-rid':
                rotation_id = list[i+1]
                skip_next_arg = True
                continue # to next arg

            if arg == '-end':
                end_age = list[i+1]
                skip_next_arg = True
                continue # to next arg

            if arg == '-begin':
                begin_age = list[i+1]
                skip_next_arg = True
                continue # to next arg

            # else, process the arg as an input file
            in_file_name = arg
            out_file_name = arg + ".gpml"
            print 
            print 'input  file_name=', in_file_name
            print 'output file_name=', out_file_name

            features = []
            features = read_dot_xy( in_file_name )

            write_features_as_gpml(features, out_file_name)
# =========================================================================
# =========================================================================
# END
