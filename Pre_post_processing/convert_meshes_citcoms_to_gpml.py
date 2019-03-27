#!/usr/bin/env python
#=====================================================================
#
#       Python Scripts for Geodynamics pre- and post- processing
#                  ---------------------------------
#
#                              Authors:
#                            Mark Turner
#          (c) California Institute of Technology 2009
#
#               Free for non-commercial academic use ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#
# This is a script to convert mesh files into GPML 
# see end of script for usage message
#

import sys, string, os, math
import numpy as np

#
# global variables  ; some will change as meshes are processed.
#

gpml_header = '''<?xml version="1.0" encoding="UTF-8"?>
<gpml:FeatureCollection xmlns:gpml="http://www.gplates.org/gplates" xmlns:gml="http://www.opengis.net/gml" xmlns:xsi="http://www.w3.org/XMLSchema-instance" gpml:version="1.6" xsi:schemaLocation="http://www.gplates.org/gplates ../xsd/gpml.xsd http://www.opengis.net/gml ../../../gml/current/base">
    <gml:featureMember>
        <gpml:MeshNode>
            <gpml:identity>GPlates-3c32b148-0215-4b30-b551-9d349aaeda0d</gpml:identity>
            <gpml:revision>GPlates-44d134e9-2ae8-4c3e-8e4b-d290994273f2</gpml:revision>
            <gpml:meshPoints>
                <gml:MultiPoint>
'''

gpml_points = ' '

## test data
gpml_test_points='''
                    <gml:pointMember>
                        <gml:Point>
                            <gml:pos>16.4998 -13.7483</gml:pos>
                        </gml:Point>
                    </gml:pointMember>
                    <gml:pointMember>
                        <gml:Point>
                            <gml:pos>16.8949 13.7773</gml:pos>
                        </gml:Point>
                    </gml:pointMember>
                    <gml:pointMember>
                        <gml:Point>
                            <gml:pos>-13.7554 14.7735</gml:pos>
                        </gml:Point>
                    </gml:pointMember>
                    <gml:pointMember>
                        <gml:Point>
                            <gml:pos>-12.9775 -9.95302</gml:pos>
                        </gml:Point>
                    </gml:pointMember>
'''

gpml_post_multipoint='''
                </gml:MultiPoint>
            </gpml:meshPoints>
            <gpml:reconstructionPlateId>
                <gpml:ConstantValue>
                    <gpml:value>0</gpml:value>
                    <gpml:valueType xmlns:gpml="http://www.gplates.org/gplates">gpml:plateId</gpml:valueType>
                </gpml:ConstantValue>
            </gpml:reconstructionPlateId>
            <gml:validTime>
                <gml:TimePeriod>
                    <gml:begin>
                        <gml:TimeInstant>
                            <gml:timePosition gml:frame="http://gplates.org/TRS/flat">http://gplates.org/times/distantPast</gml:timePosition>
                        </gml:TimeInstant>
                    </gml:begin>
                    <gml:end>
                        <gml:TimeInstant>
                            <gml:timePosition gml:frame="http://gplates.org/TRS/flat">http://gplates.org/times/distantFuture</gml:timePosition>
                        </gml:TimeInstant>
                    </gml:end>
                </gml:TimePeriod>
            </gml:validTime>
'''

## NOTE: 'sample mesh' will be replaced with out_file name for non-test runs
gpml_name='''
            <gml:name>sample mesh</gml:name>
'''

gpml_footer='''
        </gpml:MeshNode>
    </gml:featureMember>
</gpml:FeatureCollection>
'''

points = []

#=====================================================================
#=====================================================================
def convert_points_to_gpml():
    global gpml_points
    global points

    list = []

    for point in points:
        # adjust global data
        lat = point[0]
        lon = point[1]
        point ='''
                            <gml:pointMember>
                                <gml:Point>
                                    <gml:pos>%(lat)f %(lon)f</gml:pos>
                                </gml:Point>
                            </gml:pointMember>\n''' % vars()
        list.append( point )

    gpml_points = ''.join( list )
    return
#=====================================================================
#=====================================================================
def write_gpml(file):
    file.write('%s' % gpml_header)
    file.write('%s' % gpml_points)
    file.write('%s' % gpml_post_multipoint)
    file.write('%s' % gpml_name)
    file.write('%s' % gpml_footer)
#=====================================================================
#=====================================================================

#=====================================================================
def process_test():
    
    global points

    in_name = sys.argv[2]
    out_name = sys.argv[3]

    if not out_name.endswith(".gpml"):
        out_name += '.gpml'

    print ("process_test: in_file = ", in_name)
    print ("process_test: out_file = ", out_name)

    out = open(out_name, 'w')

    # simulated read of file 
    points = [ [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0], [-15.0, -15.0] ]

    convert_points_to_gpml()
    write_gpml( out )
#=====================================================================
#=====================================================================

#=====================================================================
#=====================================================================
def process_citcoms_regional():

    global points

    in_name = sys.argv[2]
    out_name = sys.argv[3]

    if not out_name.endswith(".gpml"):
        out_name += '.gpml'

    print ("process_citcoms_regional: in_file = ", in_name)
    print ("process_citcoms_regional: out_file = ", out_name)

    file_out = open(out_name, 'w')

    # read input file 
    file_in = open(in_name)
    try:
        lines = file_in.readlines()
    finally:
        file_in.close()

    # working lists to fill
    working_list = []
    theta_list = []
    phi_list = []

    lat_list = []
    lon_list = []

    # pop first header line
    lines  = lines[1:]

    # loop over the lines in the file
    for index, line in enumerate(lines):
        
        # fill the theta list
        if not 'nsd' in line: 
            working_list.append( line )

        # check for next header block
        if 'nsd= 2' in line:
            
            # copy list to theta_list
            theta_list = working_list

            # clear the working list
            working_list = []
            continue # to next line
       
       # check for final header block
        if 'nsd= 3' in line:

            # copy list to theta_list
            phi_list = working_list

            # clear the working list
            working_list = []

            break # out the loop over lines in the file

#    # loop over theta and phi and main lat lon points 
#
#    for theta_line in theta_list:
#
#        (x, theta) = theta_line.split()
#        lat = 90 - np.degrees( float(theta) )
# 
#        for phi_line in phi_list:
#
#            (y, phi) = phi_line.split()
#            lon = np.degrees( float(phi) )

    # loop over theta and phi and main lat lon points 
    for phi_line in phi_list:

        (y, phi) = phi_line.split()
        lon = np.degrees( float(phi) )

        for theta_line in theta_list:
        
            (x, theta) = theta_line.split()
            lat = 90 - np.degrees( float(theta) )
 
            print ('theta =', theta, '; lat =', lat, '; phi =', phi, ';lon =', lon)

            points.append( [lat, lon] )

    convert_points_to_gpml()

    write_gpml( file_out )

#=====================================================================
#=====================================================================
def process_citcoms_regional_cap():
    '''use a (regional) cap file to extract (regional) coordinates'''

    global points
    global gpml_name

    print ("process_citcoms_regional_cap: in_file = ")

    file_in_prefix = sys.argv[2]
    file_out_prefix = sys.argv[3]

    points = []

    in_name = "%(file_in_prefix)s" % vars()
    out_name = "%(file_out_prefix)s" % vars()
    if not out_name.endswith(".gpml"):
        out_name += '.gpml'

    print ("process_citcoms_regional_cap: in_file = ", in_name)
    print ("process_citcoms_regional_cap: out_file = ", out_name)

    # adjust feature name to match out_name 
    gpml_name = '''
        <gml:name>%(out_name)s</gml:name>
''' % vars()

    file_out = open(out_name, 'w')

    # read input file 
    file_in = open(in_name)
    try:
        lines = file_in.readlines()
    finally:
        file_in.close()

    # working lists to fill
    list = []
    theta_list = []
    phi_list = []

    # parse first line
    line1 = lines.pop(0)


    # loop over the lines in the file
    for index, line in enumerate(lines):

        cols = line.split()
        theta = cols[0]
        phi = cols[1]
        radius = cols[2]

        #print ("t,p,r = ", theta, phi, radius)

        if ( float(radius) != 1 ):
            continue # to next line

        lat = 90 - math.degrees( float(theta) )

        lon = math.degrees( float(phi) )

        points.append( [lat, lon] )

        # end of loop over lines in file 

    convert_points_to_gpml()

    write_gpml( file_out )

    print(' ')

#=====================================================================
#=====================================================================
def process_citcoms_global():

    global points
    global gpml_name

    print ("process_citcoms_global: in_file = ")

    file_in_prefix = sys.argv[2]
    file_out_prefix = sys.argv[3]

    for cap_num in [0,1,2,3,4,5,6,7,8,9,10,11]:

        points = []

        in_name = "%(file_in_prefix)s%(cap_num)d" % vars()
        out_name = "%(file_out_prefix)s%(cap_num)d" % vars()
        if not out_name.endswith(".gpml"):
            out_name += '.gpml'

        print ("process_citcoms_global: in_file = ", in_name)
        print ("process_citcoms_global: out_file = ", out_name)

        # adjust feature name to match out_name 
        gpml_name = '''
            <gml:name>%(out_name)s</gml:name>
''' % vars()

        file_out = open(out_name, 'w')

        # read input file 
        file_in = open(in_name)
        try:
            lines = file_in.readlines()
        finally:
            file_in.close()

        # working lists to fill
        list = []
        theta_list = []
        phi_list = []

        # parse first line
        line1 = lines.pop(0)
        [junk1, num_lines] = line1.split()


        # loop over the lines in the file
        for index, line in enumerate(lines):

            [theta, phi, radius] = line.split()

            #print ("t,p,r = ", theta, phi, radius)

            if ( float(radius) != 1 ):
                continue # to next line

            lat = 90 - math.degrees( float(theta) )

            lon = math.degrees( float(phi) )

            points.append( [lat, lon] )

        # end of loop over lines in file 
    
        convert_points_to_gpml()

        write_gpml( file_out )

        print ('')
    # end of loop over cap number

#=====================================================================
#=====================================================================

#=====================================================================
#=====================================================================
def usage():
    print ("" )
    print ("usage: ")
    print ("" )
    print (sys.argv[0], " format_name  in_file  out_file")
    print ("" )
    print ("format_name must be one of: citcoms_regional, citcoms_global,")
    print (" or citcoms_regional_cap")
    print ("" )
    print ("in_file is the input mesh file")
    print ("[or file prefix if citcoms_global, cap number (0 - 11) will be added]")
    print ("" )
    print ("out_file is the output file in GPML format")
    print ("[or file prefix if citcoms_global, cap number (0 - 11) will be added]")
    print ("" )
    print ("This program is intended to be run in the same directory as the input files." )
    print ("" )
    print ("Examples:" )
    print ("The global case:" )
    print ("$ convert_meshes_citcoms_to_gpml.py citcoms_global 129.coord. 129.mesh." )
    print ("" )
    print ("The regional case:" )
    print ("$ convert_meshes_citcoms_to_gpml.py citcoms_regional coor.dat 0.mesh.0.gpml")
    print ("" )
    print ("" )
#=====================================================================
#=====================================================================
if __name__ == "__main__":

    if len(sys.argv) != 4:
        usage()
        sys.exit(-1)

    format = sys.argv[1]

    if format == 'citcoms_regional':
        process_citcoms_regional()

    elif format == 'citcoms_regional_cap':
        process_citcoms_regional_cap()

    elif format == 'citcoms_global':
        process_citcoms_global()

    elif format == 'test':
        process_test()

    else :
        usage()
        sys.exit(-1)
#=====================================================================
#=====================================================================

