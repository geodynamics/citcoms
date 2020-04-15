#!/usr/bin/env python3.3
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''Core_GPlates.py has functions to convert data into GPlates forms (.rot, .gpml)'''
#=====================================================================
# This module holds data and functions for processing GPlates data.
#=====================================================================
#=====================================================================
import copy, datetime, os, pprint, re, subprocess, string, sys, traceback, uuid
import numpy as np
import random
import xml.etree.ElementTree as et

import json

# Caltech geodynamic framework modules:
import Core_Util
from Core_Util import now
import Core_GMT

#=====================================================================
# We might need to locate or install this module for the GPS system 
# if we need to read data base .dbf files
#sys.path.append('/usr/local/lib/python2.5/site-packages/dbfpy')
#from dbf import *
#=====================================================================

#=====================================================================
# Global variables
verbose = False 
verbose = True 

# FIXME:
GPGIM_VERSION = '1.6.0329'

#====================================================================
#====================================================================
# XML and GPML related variables, and initializations 

# Set and Register the namespaces
ns = {'gml' : 'http://www.opengis.net/gml', 
      'xsi' : 'http://www.w3.org/XMLSchema-instance',
     'gpml' : 'http://www.gplates.org/gplates', 
}
for n in ns: et.register_namespace(n, ns[n])

#====================================================================
#====================================================================
def displacement_to_euler_pole( lat1, lon1, lat2, lon2 ):
    '''Compute the great circle path Euler Pole to move p1 to p2'''

    # Unit Sphere to compute angles and positions, it works 
    v1 = np.array( Core_Util.sph2cart(lat1, lon1, 1.0) )
    v2 = np.array( Core_Util.sph2cart(lat2, lon2, 1.0) )

    #if verbose: print( 'v1 = ', v1, '; v2 = ', v2 )

    # Get the inner product of the position vectors 
    dot_p = np.dot(v1, v2) 

    #if verbose: print( 'dot_p = ', dot_p )

    # Compute the angle between the position vectors 
    pole_ang = np.degrees( np.arccos( dot_p ) )

    # Get the cross product of the position vectors
    cross_p = np.cross(v1, v2)
    cross_p = cross_p / np.linalg.norm( cross_p )

    # Set the Euler pole location
    pole_lon = np.degrees( np.arctan2( cross_p[0], cross_p[1]) )
    pole_lon = 90-pole_lon

    pole_lat = np.degrees( np.arcsin( cross_p[2]) )

    # correct for indeterminate cases:
    if np.isnan( pole_ang ) : pole_ang = 0.0
    if np.isnan( pole_lat ) : pole_lat = 0.0
    if np.isnan( pole_lon ) : pole_lon = 0.0

    if verbose: 
        print( 'pole_lat =', pole_lat, '; pole_lon =', pole_lon, '; pole_ang=', pole_ang)

    return pole_lat, pole_lon, pole_ang
# =========================================================================
# =========================================================================
def read_dot_rot( file_name ):
    '''Read a .rot file into a dictionary with keys:
moving_id, reconstruction_age, lat, lon, angle, fixed_id, comment; '''

    # list of rotations to return
    rotations = []

    if verbose: print( 'read_dot_rot: open:', file_name )
    file = open( file_name )
    try:
        lines = [ line.rstrip() for line in file if line[0] != '#' ]
    finally:
        file.close()

    for l in lines:

        rot_list = l.split()

        moving_id = rot_list[0]
        recon_age = rot_list[1]
        lat = rot_list[2]
        lon = rot_list[3] 
        angle = rot_list[4]
        fixed_id = rot_list[5]
        comment = ' '.join( rot_list[6:] )

        rot = {} 
        rot['moving_id'] = moving_id
        rot['reconstruction_age'] = recon_age
        rot['lat'] = lat
        rot['lon'] = lon
        rot['angle'] = angle
        rot['fixed_id'] = fixed_id
        rot['comment'] = comment

        rotations.append( rot )

    return rotations

# =========================================================================
# =========================================================================
def process_features_with_displacements_into_poles( mesh, fixed_id ):
    '''Convert feature dictionaries with displacement data into rotations data returned as a list of Euler Poles with rotations: (pole_lat, pole_lon, angle) '''

    # a list of Euler poles to create from feature displacement data 
    poles = []

    features = mesh['interior'] 

    # loop over features
    for f in features :

      # Set the initial data 
      name = f['properties']['name']

      # initialize vars using present day coords
      start_lat = f['properties']['geometry'][0]['lat']
      start_lon = f['properties']['geometry'][0]['lon']

      rot_id = f['properties']['rot_id']

      # process each displacement / time pair 
      for displacement in f['displacements'] :

        # parse displacement dict 
        recon_age = displacement['reconstruction_age']
        pos_lat = displacement['position']['lat']
        pos_lon = displacement['position']['lon']
        pos_ang = 0 

        # an empty rot dict to populate
        rot = {} 
        rot['lat'] = 0 
        rot['lon'] = 0 
        rot['angle'] = 0 

        # Set up initial rotation 
        rot['fixed_id'] = fixed_id
        rot['moving_id'] = rot_id 
        rot['reconstruction_age'] = recon_age

        com = '; displacement from ( %(start_lat)0.4f, %(start_lon)0.4f ) to ( %(pos_lat)0.4f , %(pos_lon)0.4f )' % vars()
        rot['comment'] = '! Rotation for ' + name + com

        # set up a default pole
        pole = [0,0,0]

        # compute the Euler Pole and rotation
        pole = displacement_to_euler_pole( start_lat, start_lon, pos_lat, pos_lon)

        # update rot with components
        rot['lat']   = pole[0]
        rot['lon']   = pole[1]
        rot['angle'] = pole[2]

        poles.append( rot )

    # end of loop over displacement

    return poles

# =========================================================================
# =========================================================================
def print_rotations_as_dot_rot( rotations): 
    ''' print a rotation dictionary in the .rot format'''
    for rot in rotations:
        s = '' + \
        ('%4d'   % int(   rot['moving_id'] )) + ' ' + \
        ('%6.1f' % float( rot['reconstruction_age'] )     ) + ' ' + \
        ('%6.2f' % float( rot['lat'] )      ) + ' ' + \
        ('%6.2f' % float( rot['lon'] )      ) + ' ' + \
        ('%6.2f' % float( rot['angle'] )    ) + ' ' + \
        ('%4d'   % int(   rot['fixed_id'] ) ) + ' ' + \
        ('%s'    % str(rot['comment'] )     ) + ' '
        print( s )
# =========================================================================
def write_rotations_as_dot_rot( rotations,  out_file_name ):
    ''' output a rotation dictionary as a .rot file'''
    if verbose: print (Core_Util.now(), 'write_rotations_as_dot_rot: open:', out_file_name )
    out_file = open( out_file_name, 'w')
    for rot in rotations:
        s = \
        ('%4d'   % int(   rot['moving_id'] )) + ' ' + \
        ('%6.1f' % float( rot['reconstruction_age'] )     ) + ' ' + \
        ('%6.2f' % float( rot['lat'] )      ) + ' ' + \
        ('%6.2f' % float( rot['lon'] )      ) + ' ' + \
        ('%6.2f' % float( rot['angle'] )    ) + ' ' + \
        ('%4d'   % int(   rot['fixed_id'] ) ) + ' ' + \
        ('%s'    % str(rot['comment'] )     ) + ' ' + \
        '\n'
        out_file.write( s )
    out_file.close()
# =========================================================================
# =========================================================================

# =========================================================================
# =========================================================================
# SAVE this ElementTree code for reference: 
    # 1. and 2. are the same since FeatureCollection only has featuremembers

    # 1.
    #for f in root.findall("./*", namespaces=ns):
    #    print ('FOUND!')
    #    print('f.tag =', f.tag)
 
    # 2.
    # for f in root.findall("./gml:featureMember", namespaces=ns):
    #     print ('FOUND!')
    #     print('f.tag =', f.tag)

    #for f in root.findall('featureMember', ns):

    # change a poperty
    #for e in root.findall('./gml:featureMember/gpml:TopologicalNetwork/gpml:networkShapeFactor', namespaces=ns):
    #    print ('e.tag=', e.tag)
    #    e.text = '999999'

    #for e in root.findall('./gml:featureMember', namespaces=ns):
    #    print ('e.tag=', e.tag)
    #    root.remove( e )
#=====================================================================
#=====================================================================

#=====================================================================
#=====================================================================
def transform_gpml_file_network_to_polygon( file_name, dir_prefix='./'):
    '''Use ElementTree to parse a .gpml file and transform all gpml:TopologicalNetwork to gpml:TopologicalClosedPlateBoundary features.'''
 
    print (now(), 'transform_gpml_file_network_to_polygon: START' )
    print (now(), 'transform_gpml_file_network_to_polygon: dir_prefix =', dir_prefix )
    print (now(), 'transform_gpml_file_network_to_polygon: file_name =', file_name )

    # get the xml 
    tree = et.parse( file_name )
    root = tree.getroot()

    # Loop over all network features
    for feature in root.findall('./gml:featureMember', namespaces=ns) :

        ftype = feature.find('./*', namespaces=ns) 

        print ('')
        print (now(), 'transform_gpml_file_network_to_polygon: found feature type =', ftype.tag )

        # Get the gpml:TopologicalNetwork feature tag
        tn = feature.find('./gpml:TopologicalNetwork', namespaces=ns)

        # Only process gpml:TopologicalNetwork features
        if tn == None: 
            continue # to next feature 

        # update the identity 
        id_tag = tn.find('./gpml:identity', namespaces=ns)
        old_id = id_tag.text 
        new_id = old_id + '_transformed_by_net2polygon'
        id_tag.text = new_id
        id_tag.set('updated_by', 'Core_GPlates.transform_gpml_file_network_to_polygon')
        print (now(), 'transform_gpml_file_network_to_polygon:             old id =', old_id )
        print (now(), 'transform_gpml_file_network_to_polygon:             new id =', tn.find('./gpml:identity', namespaces=ns).text )
       
        # check for optional name tag 
        old_name = ''
        new_name = ''
        name = feature.find('./*/gml:name', namespaces=ns) 
        if name != None : 
            old_name = name.text
            print (now(), 'transform_gpml_file_network_to_polygon:           old name =', old_name )
        else :
            print (now(), 'transform_gpml_file_network_to_polygon: WARNING gml:name property missing; creating tag.')
            name = et.SubElement(tn, 'gml:name')
            old_name = 'gml:name property missing'

        # update the name 
        new_name = ''.join(['Transformed to Rigid Plate from "', old_name, '"'])
        name.text = str(new_name) 
        name.set('updated_by', 'Core_GPlates.transform_gpml_file_network_to_polygon')
        print (now(), 'transform_gpml_file_network_to_polygon:           new name =', name.text )

        # remove network specific properties 
        x = tn.find('./gpml:networkShapeFactor', namespaces=ns)
        if x != None: 
            if verbose: print (now(), 'transform_gpml_file_network_to_polygon: remove tag =', x.tag)
            tn.remove(x)

        # remove network specific properties 
        x = tn.find('./gpml:networkMaxEdge', namespaces=ns)
        if x != None:
            if verbose: print (now(), 'transform_gpml_file_network_to_polygon: remove tag =', x.tag)
            tn.remove(x)

        # get the nested gpml:TopologicalNetwork element 
        tnet = tn.find('./gpml:network/gpml:ConstantValue/gpml:value/gpml:TopologicalNetwork', namespaces=ns)
        if tnet == None:
            print (now(), 'transform_gpml_file_network_to_polygon: WARNING missing nested xml element:')
            print (now(), 'transform_gpml_file_network_to_polygon: /gpml:network/gpml:ConstantValue/gpml:value/gpml:TopologicalNetwork')
            print (now(), 'transform_gpml_file_network_to_polygon: skipping this feature')
            continue # to next feature 

        # else
        print (now(), 'transform_gpml_file_network_to_polygon: found net.tag=', tnet.tag)

        # remove interior sections 
        #for i in tnet.findall('./gpml:interior', namespaces=ns) :
        #   #print (now(), 'transform_gpml_file_network_to_polygon: remove tag =', i.tag)
        #   tnet.remove(i)

        # get the boundary TopologicalSections element 
        sections = et.Element('gpml:TopologicalSections')
        sections.clear()
        sections = tnet.find('./gpml:boundary/gpml:TopologicalSections', namespaces=ns)
        print (now(), 'transform_gpml_file_network_to_polygon: tnet.find tag =', sections.tag)

        s = tnet.findall('./gpml:boundary/gpml:TopologicalSections/gpml:section', namespaces=ns)
        print (now(), 'transform_gpml_file_network_to_polygon: # of gpml:section elements =', len(s))

        #print( now(), '================================== DUMP ================================' )
        #et.dump( sections )
        #print( now(), '================================== DUMP ================================' )

        # remove the original gpml:network property 
        n = tn.find('gpml:network', namespaces=ns)
        tn.remove(n)

        # create a new replacement feature for the working gpml:TopologicalNetwork
        feature_member = et.Element('gml:featureMember')

        # create a new element for the TCPB
        tcpb = et.Element('gpml:TopologicalClosedPlateBoundary')

        # copy over all remaining properties from old feature to the TCPB
        for p in feature.findall('./gpml:TopologicalNetwork/*', namespaces=ns) :
            print (now(), 'transform_gpml_file_network_to_polygon: copy property tag =', p.tag)
            tcpb.append(p)

        # create a new boundary element from geodynamic_framework_data file
        gpml_boundary_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_boundary.gpml'
        gpml_boundary_tree = et.parse( gpml_boundary_file )
        gpml_boundary_root = gpml_boundary_tree.getroot()
        boundary = gpml_boundary_root
        print (now(), 'transform_gpml_file_network_to_polygon: boundary.tag =', boundary.tag)

        # get the gpml:exterior node 
        exterior = boundary.find('./gpml:PiecewiseAggregation/gpml:timeWindow/gpml:TimeWindow/gpml:timeDependentPropertyValue/gpml:ConstantValue/gpml:value/gpml:TopologicalPolygon/gpml:exterior', namespaces=ns)
        print (now(), 'transform_gpml_file_network_to_polygon: found exterior.tag =', exterior.tag)

        for i in list(exterior) : 
            print( now(), 'transform_gpml_file_network_to_polygon:   FIND: exterior children i.tag =', i.tag)

        # clear the old exterior
        exterior.clear()
        for i in list(exterior) : 
            print( now(), 'transform_gpml_file_network_to_polygon:  CLEAR: exterior children i.tag =', i.tag)

        # add the boundary sections to the exterior node 
        exterior.append(sections)
        for i in list(exterior) : 
            print( now(), 'transform_gpml_file_network_to_polygon: APPEND: exterior children i.tag =', i.tag)
        
        # Add the boundary 
        tcpb.append(boundary) 

        # add the TCPB to the feature
        feature_member.append(tcpb) 

        # append the new gpml:TopologicalClosedPlateBoundary feature
        root.append( feature_member )

        # remove the old gpml:TopologicalNetwork feature
        root.remove( feature )

    # end of loop over features

    # create the new file 
    new_file_name = dir_prefix + 'TRANSFORMED_net2polygon.' + file_name.replace('.gpml', '.gpml')
    print( now(), 'transform_gpml_file_network_to_polygon: open:', new_file_name )

    # write out the whole tree as xml to a new file 
    tree.write(new_file_name, encoding="UTF-8", xml_declaration='<?xml version="1.0" encoding="UTF-8"?>', default_namespace=None, method="xml")
    
#=====================================================================
#=====================================================================
def batch_transform_net_to_polygon():
    '''Batch tool to read a set of .gpml files and create a companion set,
where all TopologicalNetworks are transformed to TopologicalClosedPlateBoundary feature types.  
Run with this command:
  $ Core_GPlates -net2polygon *.gpml
Each input .gpml file will be processed and given a new file name prefix of 'TRANSFORMED_net2polygon.',
and placed in a new directory: ./Transformed_Networks_to_Platepolygons/ '''

    file_list = sys.argv[2:]

    dir_prefix = './Transformed_Networks_to_Platepolygons/'
    Core_Util.make_dir( dir_prefix ) 

    for filename in file_list :
        transform_gpml_file_network_to_polygon(filename, dir_prefix) 

#=====================================================================
#=====================================================================
def batch_transform_iodpkml_to_gpml():
    '''turn a set of iodpkml files to gpml files'''
    file_list = sys.argv[2:]
    print( now(), 'batch_transform_iodpkml_to_gpml: file_list =', file_list )

    for in_file_name in file_list :

        # empty list of features to create 
        features = []

        # First get a list of layers from the .kml file 
        cmd = 'ogrinfo ' + in_file_name 
        print( now(), 'batch_transform_iodpkml_to_gpml:', cmd )
        out = subprocess.check_output( cmd, shell=True, universal_newlines=True)
        #print( now(), 'batch_transform_iodpkml_to_gpml: out = \n', out )

        # split the string on new line 
        out_lines = out.split('\n')
        #print( now(), 'out_lines = ', out_lines )

        # check output for layer number 

        # select all the layers from the output lines 
        pattern = re.compile('^\d+:')

        # TEST: select only one layer 
        #pattern = re.compile('^1:')

        for line in out_lines :
            match = pattern.match( line )
            if match:
                # get layer name
                l = line.split()
                layer_name = l[1] + ' ' + l[2]
                print( now(), '\nbatch_transform_iodpkml_to_gpml: processing layer:', layer_name)

                # Get data for this layer 
                cmd = 'ogrinfo ' + in_file_name + ' "' + layer_name + '"'
                print( now(), 'batch_transform_iodpkml_to_gpml:', cmd )
                layer_data = subprocess.check_output( cmd, shell=True, universal_newlines=True)
                #print( now(), 'batch_transform_iodpkml_to_gpml: layer_data = \n', layer_data )

                # add the features from this layer to the main collection
                features += iodpkml_string_to_gpml_features(layer_data) 

            # else, this is not a feature block 

        if verbose : 
            print( now(), 'batch_transform_iodpkml_to_gpml: print a single feature to show form:')
            Core_Util.tree_print( features[0] )

        # create the feature collection
        new_file_name = in_file_name.replace('.kml', '.gpml')

        # write out the gpml file 
        create_gpml_feature_collection_from_list_of_features( new_file_name, features )

    # end of loop over input iodpkml files

#=====================================================================
#=====================================================================
def iodpml_desc_string_to_property_dict(s):
    '''Parse an IODP KML Description html string into a dictionary of feature properties'''

    # a dict of props to return
    props = {}

    # split on the rows of the table 
    rows = s.split('<tr>')

    #print( now(), 'iodpml_desc_string_to_property_dict: rows =')
    #Core_Util.tree_print( rows ) 

    # Skip first few header rows
    for r in rows[2:] :

        # skip mid-table header row 
        if "<b>Data</b>" in r : continue # to next row

        r = r.strip()
        r = r.rstrip()
        #print( now(), 'iodpml_desc_string_to_property_dict: r =', r)

        parser = Core_Util.iodp_table_parser()
        parser.feed( r )
        l = parser.get_data()
        del parser

        if len(l) == 1:
            l.append('Unknown')

        # print( now(), 'iodpml_desc_string_to_property_dict: l =', l)

        # make property value pair for this row
        props[ str( l[0] ).replace(':','') ] = l[1] 

    return props
#=====================================================================
#=====================================================================
def iodpkml_string_to_gpml_features( string ):
    '''process an iodpkml layer string (with multiple features) into gpml features'''

    # empty list of features to create 
    features = []

    # start up an warning file 
    warn_file = open('iodpkml2gpml_warnings.txt', 'w')

    # process the layer string 
    f_list = string.split('OGRFeature')

    # remove all the header lines
    f_list = f_list[1:]

    for i,f in enumerate( f_list ):

        print(' ')

        # get the general feature dict 
        feature = copy.deepcopy( get_base_feature() )

        # Set type 
        feature['type'] = 'gpml:OceanDrillSite'

        f_lines = f.split('\n')

        # extract certain lines and remove whitespace
        name_line = f_lines[1]
        header,name = name_line.split('=')
        name = name.strip()
        name = name.rstrip()

        print( now(), 'iodpkml_string_to_gpml_features: feature', i, ' name =', name)

        # update the feature dict
        feature['properties']['name'] = name 
        feature['properties']['description'] = name 

        # update the feature dict with default values
        feature['properties']['begin_age'] = 250 
        feature['properties']['end_age'] = 0

        feature['properties']['depthToBasement'] = 0.0
        feature['properties']['sedimentThickness'] = 0.0
        feature['properties']['waterDepth'] = 0.0
        feature['properties']['coreRecovered'] = 0.0

        feature['properties']['expedition'] = 'Unknown'
        feature['properties']['site'] = 'Unknown'
        feature['properties']['hole'] = 'Unknown'
        feature['properties']['program'] = 'Unknown'

        feature['properties']['logData'] = 'Unknown'
        feature['properties']['publication'] = 'Unknown'
        feature['properties']['initialReportVolume'] = 'Unknown'
        feature['properties']['coreData'] = 'Unknown'

        # process the html table into props 
        desc_line = f_lines[2]
        prop_d = iodpml_desc_string_to_property_dict( desc_line )

        for prop in prop_d: 

            value = prop_d[prop]

            if prop == 'Longitude'  : continue # to next prop
            if prop == 'Latitude'   : continue # to next prop 

            if   prop == 'Expedition'           : key = 'expedition'
            elif prop == 'Site'                 : key = 'site'
            elif prop == 'Hole'                 : key = 'hole'
            elif prop == 'Program'              : key = 'program'

            elif prop == 'Water Depth'          : 
              key = 'waterDepth'
              value = value.replace(' m','')
              value = float( value ) 

            elif prop == 'Log Data'             : key = 'logData'
            elif prop == 'Publication'          : key = 'publication'
            elif prop == 'Initial Report Volume': key = 'initialReportVolume'
            elif prop == 'Core Data'            : key = 'coreData'

            elif prop == 'Core Recovered'       : 
              key = 'coreRecovered'
              value = value.replace(' m','')
              value = float( value ) 
            else : continue # to next prop 

            # print(now(), 'iodpkml_string_to_gpml_features: key =', key, '; value =', prop_d[prop]) 

            feature['properties'][key] = value 

        # extract the coords 
        geom_line = f_lines[3]
        geom_line = geom_line.replace('  POINT (', '')
        geom_line = geom_line.replace(')', '')
        coords = geom_line.split()
        lon = float( coords[0] )
        lat = float( coords[1] )

        # set geometry data
        g_type = 'Point'
        coords = [lat, lon]
        feature['geometry']['type'] = g_type
        feature['geometry']['coordinates'] = coords

        # get age data from grid 
        # write tmp xy file 
        xy_data = '%(lon)f %(lat)f' % vars()
        xy_name = './' + name.replace(' ','_') + '.xy'
        xy_file = open(xy_name, 'w')
        xy_file.write(xy_data)
        xy_file.close()

        age_grid = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'agegrid_final_nomask_0.grd'
        cmd = xy_name + ' -G' + age_grid
        age = Core_GMT.callgmt( 'grdtrack', cmd, {'Z':''}, '2>', '/dev/null' ) # send gmt stdwarn to /dev/null
        #print(now(), 'iodpkml_string_to_gpml_features: age =', age)

        # clean up tmp file 
        os.remove( xy_name )

        # set the age data
        feature['properties']['end_age'] = 0
        feature['properties']['begin_age'] = float( age )

        # check for missing data 
        if age == 'NaN':
            print( now(), 'iodpkml_string_to_gpml_features: WANRING: age not found from grid; setting to 200 Ma')
            feature['properties']['begin_age'] = float( 200 )
            warn_file.write('Feature ' + name + 'missing begin_age')

        print(now(), 'iodpkml_string_to_gpml_features: feature properties =')
        Core_Util.tree_print( feature['properties'] ) 

        # Append the feature to the list
        features.append( feature ) 
    
    return features
#=====================================================================
#=====================================================================
#=====================================================================
def batch_transform_json_to_gpml():
    '''turn a set of json files to gpml files'''
    file_list = sys.argv[2:]
    for json_file_name in file_list :
        json_file = open(json_file_name, 'r')
        json_string = json_file.readline()
        print( now(), 'batch_transform_json_to_gpml: json_string =', json_string )
        json_to_gpml(json_file_name.replace('.json', '.gpml'), json_string)
#=====================================================================
#=====================================================================
def batch_transform_txt_to_gpml():
    '''turn a set of txt files to gpml files'''
    file_list = sys.argv[2:]
    for txt_file_name in file_list :
        txt_to_gpml( txt_file_name ) 
#=====================================================================
#=====================================================================
#=====================================================================
def create_displacement_feature_collections( mesh ) :
    '''From a set of point data representing a topological network, create two feature colllections (.gpml files):
one collection will hold all the points as gpml:DisplacementPoint features,
one collection will hold a single gpml:TopologicalNetwork feature, 
linking to the the boundary and interior features.'''

    # Read gpml files from geodynamic_framework_data/

    # a feature collection to hold the displacement points
    points_fc_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_feature_collection.gpml'
    points_fc_tree = et.parse( points_fc_file )
    points_fc_root = points_fc_tree.getroot()
    if verbose: print (now(), 'create_displacement_feature_collections: points_fc_root.tag =', points_fc_root.tag)

    # a feature collection to hold the topology network feature 
    network_fc_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_topology_network.gpml'
    network_fc_tree = et.parse( network_fc_file )
    network_fc_root = network_fc_tree.getroot()
    if verbose: print (now(), 'create_displacement_feature_collections: network_fc_root.tag =', network_fc_root.tag)

    # get the gpml:TopologicalNetwork element
    topological_network = network_fc_root.find('./gml:featureMember/gpml:TopologicalNetwork/gpml:network/gpml:ConstantValue/gpml:value/gpml:TopologicalNetwork' , namespaces=ns)

    # save for later
    # update the valueType element
    #vt = network_fc_root.find('./gml:featureMember/gpml:TopologicalNetwork/gpml:network/gpml:ConstantValue/gpml:valueType', namespaces=ns)
    #vt.attrib = {'xmlns:gpml', '"http://www.gplates.org/gplates"'}

    boundary_features = mesh['boundary'] 

    # loop over boundary features
    for i,f in enumerate( boundary_features )  :

        #Core_Util.tree_print( f ) 

        # create a unique feature id for this feature
        feature_id = 'GPlates-' + str(uuid.uuid4()) 

        # read the sample point 
        point_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_point.gpml'
        point_tree = et.parse( point_file )
        point_root = point_tree.getroot()
    
        # clear out the attibs on this element  
        point_root.attrib = {}

        # find and replace property values 
        fid = point_root.find('./gpml:identity', namespaces=ns)
        fid.text = feature_id 

        name = point_root.find('./gml:name', namespaces=ns)
        name.text = f['properties']['name']

        desc = point_root.find('./gml:description', namespaces=ns)
        desc.text = f['properties']['description']

        begin = point_root.find('./gml:validTime/gml:TimePeriod/gml:begin/gml:TimeInstant/gml:timePosition', namespaces=ns)
        begin.text = str ( f['properties']['begin_age'] )
 
        end = point_root.find('./gml:validTime/gml:TimePeriod/gml:end/gml:TimeInstant/gml:timePosition', namespaces=ns)
        end.text = str( f['properties']['end_age'] )

        # process geometry 
        lat = f['properties']['geometry'][0]['lat']
        lon = f['properties']['geometry'][0]['lon']
        pos = point_root.find('./gpml:position/gml:Point/gml:pos', namespaces=ns)
        pos.text = str( str(lat) + ' ' + str(lon) )

        rid = f['properties']['rot_id'] 
        rot_id = point_root.find('./gpml:reconstructionPlateId/gpml:ConstantValue/gpml:value', namespaces=ns)
        rot_id.text = str( rid )
 
        # create a new feature member element for the working feature 
        feature_member = et.Element('gml:featureMember')
        # append the feature to the new points_fc featureMember element 
        feature_member.append( point_root )
        # append the FeatureMember element to the displacement points feature collection
        points_fc_root.append( feature_member )

        # get the gpml:TopologicalSections element 
        sections = topological_network.find('./gpml:boundary/gpml:TopologicalSections', namespaces=ns)

        # create a gpml:section element and sub elements  for this working boundary feature
        section = et.Element('gpml:section')
        topo_pnt = et.SubElement( section, 'gpml:TopologicalPoint')
        src_geom = et.SubElement( topo_pnt, 'gpml:sourceGeometry')
        prop_del = et.SubElement( src_geom, 'gpml:PropertyDelegate')
        target_f = et.SubElement( prop_del, 'gpml:targetFeature')
        target_f.text = feature_id
        target_p = et.SubElement( prop_del, 'gpml:targetProperty', {'xmlns:gpml': "http://www.gplates.org/gplates"} )
        target_p.text = 'gpml:position'
        value_type = et.SubElement( prop_del, 'gpml:valueType', {'xmlns:gml': "http://www.opengis.net/gml"} )
        value_type.text = 'gml:Point'

        # append this gpml:section element to the gpml:TopologicalSections element
        sections.append( section )

    # end of loop over boundary features

    # loop over interior features 
    for i,f in enumerate( mesh['interior'] )  :

        # create a unique feature id for this feature
        feature_id = 'GPlates-' + str(uuid.uuid4()) 

        # read the sample point 
        point_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_point.gpml'
        point_tree = et.parse( point_file )
        point_root = point_tree.getroot()
    
        # clear out the attibs on this element  
        point_root.attrib = {}

        # find and replace property values 
        fid = point_root.find('./gpml:identity', namespaces=ns)
        fid.text = feature_id 

        name = point_root.find('./gml:name', namespaces=ns)
        name.text = f['properties']['name']

        desc = point_root.find('./gml:description', namespaces=ns)
        desc.text = f['properties']['description']

        begin = point_root.find('./gml:validTime/gml:TimePeriod/gml:begin/gml:TimeInstant/gml:timePosition', namespaces=ns)
        begin.text = str ( f['properties']['begin_age'] )
 
        end = point_root.find('./gml:validTime/gml:TimePeriod/gml:end/gml:TimeInstant/gml:timePosition', namespaces=ns)
        end.text = str( f['properties']['end_age'] )

        # process geometry 
        lat = f['properties']['geometry'][0]['lat']
        lon = f['properties']['geometry'][0]['lon']
        pos = point_root.find('./gpml:position/gml:Point/gml:pos', namespaces=ns)
        pos.text = str( str(lat) + ' ' + str(lon) )

        rid = f['properties']['rot_id'] 
        rot_id = point_root.find('./gpml:reconstructionPlateId/gpml:ConstantValue/gpml:value', namespaces=ns)
        rot_id.text = str( rid )
 
        # create a new feature member element for the working feature 
        feature_member = et.Element('gml:featureMember')
        # append the feature to the new points_fc featureMember element 
        feature_member.append( point_root )
        # append the FeatureMember element to the displacement points feature collection
        points_fc_root.append( feature_member )

        # create a gpml:interior element, and sub elements, for this working feature 
        interior = et.Element('gpml:interior')
        tnet_int = et.SubElement( interior, 'gpml:TopologicalNetworkInterior')
        src_geom = et.SubElement( tnet_int, 'gpml:sourceGeometry')
        prop_del = et.SubElement( src_geom, 'gpml:PropertyDelegate')

        target_f = et.SubElement( prop_del, 'gpml:targetFeature')
        target_f.text = feature_id

        target_p = et.SubElement( prop_del, 'gpml:targetProperty', {'xmlns:gpml': "http://www.gplates.org/gplates"} )
        target_p.text = 'gpml:position'

        value_type = et.SubElement( prop_del, 'gpml:valueType', {'xmlns:gml': "http://www.opengis.net/gml"} )
        value_type.text = 'gml:Point'

        # append this interior element tree to the gpml:TopologicalNetwork element
        topological_network.append( interior )

    # end of loop on features

    #print( now(), '================================== DUMP ====================================' )
    #et.dump( points_fc_root )
    #print( now(), '================================== DUMP ====================================' )

    # create the new files

    # write out the displacement points fc
    points_fc_file_name = mesh['file_name_prefix'] + '.displacement_points.gpml'
    print( now(), 'create_displacement_feature_collections: open:', points_fc_file_name )
    points_fc_tree.write(points_fc_file_name, encoding="UTF-8", \
        xml_declaration='<?xml version="1.0" encoding="UTF-8"?>', \
        default_namespace=None, method="xml")

    # write out the network fc
    network_fc_file_name = mesh['file_name_prefix'] + '.network.gpml'
    print( now(), 'create_displacement_feature_collections: open:', network_fc_file_name )
    network_fc_tree.write(network_fc_file_name, encoding="UTF-8", \
        xml_declaration='<?xml version="1.0" encoding="UTF-8"?>', \
        default_namespace=None, method="xml")

#=====================================================================
#=====================================================================
def create_gpml_feature_collection_from_list_of_features( file_name, feature_list ):
    '''write a new gpml file with file_name, containing the features in feature list '''

    # Read gpml files from geodynamic_framework_data/

    # a feature collection 
    fc_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_feature_collection.gpml'
    fc_tree = et.parse( fc_file )
    fc_root = fc_tree.getroot()
    #if verbose: print (now(), 'create_gpml_feature_collection_from_list_of_features: fc_root.tag =', fc_root.tag)

    # update to latest GPGIM version
    fc_root.attrib['{http://www.gplates.org/gplates}version'] = GPGIM_VERSION
  
    #if verbose: 
    #    print( now(), 'create_gpml_feature_collection_from_list_of_features: fc_root.attrib =')
    #    Core_Util.tree_print( fc_root.attrib )
    
    # loop over the features, create element trees for them, and append to the collection
    for f in feature_list:
        value = create_gpml_feature_from_feature_dictionary( f ) 
        if not value == None : fc_root.append( value ) 

    # write out the feature collection
    print( now(), 'create_gpml_feature_collection_from_list_of_features: open:', file_name )
    fc_tree.write(file_name, encoding="UTF-8", \
        xml_declaration='<?xml version="1.0" encoding="UTF-8"?>', \
        default_namespace=None, method="xml")

#=====================================================================
#=====================================================================
def create_gpml_feature_from_feature_dictionary( f ):
    '''from a feature dictionary create the '''

    # read the sample feature and set geometry based on geom type:
    feature_root = None

    if f['geometry']['type'] == 'Point':
        g_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_point.gpml'
        tree = et.parse( g_file )
        feature_root = tree.getroot()

        # set the geometry 
        pos = feature_root.find('./gpml:position/gml:Point/gml:pos', namespaces=ns)
        lon = f['geometry']['coordinates'][0]
        lat = f['geometry']['coordinates'][1]
        # set the text in the element 
        pos.text = str(lat) + ' ' + str(lon) 

    elif f['geometry']['type'] == 'MultiPoint':
        g_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_multipoint.gpml'
        tree = et.parse( g_file )
        feature_root = tree.getroot()

        # set the geometry 
        multipoint = feature_root.find('./gpml:multiPosition/gml:MultiPoint', namespaces=ns)

        # Loop over coords
        text = ''
        #print("FIXME:", f['geometry']['coordinates'] )

        for coord in f['geometry']['coordinates'] :
            lon = coord[0]
            lat = coord[1]
            text = str(lat) + ' ' + str(lon) + ' ' 

            # create a new elements for this point 
            ptm = et.Element('gml:pointMember')
            pnt = et.SubElement(ptm, 'gml:Point')
            pos = et.SubElement(pnt, 'gml:pos')
            pos.text = text

            # append the pointMember to the gml:MultiPoint
            multipoint.append( ptm )

    elif f['geometry']['type'] == 'unclassified_geom_multipoint':
        g_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_unclassified_geom_multipoint.gpml'
        tree = et.parse( g_file )
        feature_root = tree.getroot()

        # set the geometry 
        multipoint = feature_root.find('./gpml:unclassifiedGeometry/gpml:ConstantValue/gpml:value/gml:MultiPoint', namespaces=ns)

        # Loop over coords
        text = ''
        #print("FIXME:", f['geometry']['coordinates'] )

        for coord in f['geometry']['coordinates'] :
            lon = coord[0]
            lat = coord[1]
            text = str(lat) + ' ' + str(lon) + ' ' 

            # create a new elements for this point 
            ptm = et.Element('gml:pointMember')
            pnt = et.SubElement(ptm, 'gml:Point')
            pos = et.SubElement(pnt, 'gml:pos')
            pos.text = text

            # append the pointMember to the gml:MultiPoint
            multipoint.append( ptm )

    elif f['geometry']['type'] == 'LineString':
        g_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_line.gpml'
        tree = et.parse( g_file )
        feature_root = tree.getroot()

        # set the geometry 
        pos = feature_root.find('./gpml:centerLineOf/gpml:ConstantValue/gpml:value/gml:OrientableCurve/gml:baseCurve/gml:LineString/gml:posList', namespaces=ns)
        # Loop over coords
        text = ' '
        for coord in f['geometry']['coordinates'] :
            lon = coord[0]
            lat = coord[1]
            text += str(lat) + ' ' + str(lon) + ' ' 
        # set the text in the element 
        pos.text = text 

    elif f['geometry']['type'] == 'Polygon':
        g_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_polygon.gpml'
        tree = et.parse( g_file )
        feature_root = tree.getroot()

        # set the geometry 
        pos = feature_root.find('./gpml:boundary/gpml:ConstantValue/gpml:value/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList', namespaces=ns)
        # Loop over coords
        text = ' '
        for coord in f['geometry']['coordinates'] :
            lon = coord[0]
            lat = coord[1]
            text += str(lat) + ' ' + str(lon) + ' ' 
        # set the text in the element 
        pos.text = text 

    else:
        print(now(), "create_gpml_feature_from_feature_dictionary: ERROR: f['geometry']['type'] not set!" )
        return None

    # clear out the attibs on this element  
    feature_root.attrib = {}

    # print the tag 
    #print(now(), 'create_gpml_feature_from_feature_dictionary: ORIG feature_root.tag = ', feature_root.tag)

    # create a unique feature id for this feature
    feature_id = 'GPlates-' + str(uuid.uuid4()) 

    # find and replace property values 

    fid = feature_root.find('./gpml:identity', namespaces=ns)
    fid.text = feature_id 

    name = feature_root.find('./gml:name', namespaces=ns)
    if 'name' in f['properties']:
        name.text = f['properties']['name']
        del f['properties']['name']
    else :
        name.text = 'Unknown'

    desc = feature_root.find('./gml:description', namespaces=ns)
    if 'description' in f['properties'] :
        desc.text = f['properties']['description']
        del f['properties']['description']
    else:
        desc.text = ''

    begin = feature_root.find('./gml:validTime/gml:TimePeriod/gml:begin/gml:TimeInstant/gml:timePosition', namespaces=ns)
    if 'begin_age' in f['properties']:
        begin.text = str( f['properties']['begin_age'] )
        del f['properties']['begin_age']
    else: 
        begin.text = str(200)

    end = feature_root.find('./gml:validTime/gml:TimePeriod/gml:end/gml:TimeInstant/gml:timePosition', namespaces=ns)
    if 'end_age' in f['properties']:
        end.text = str( f['properties']['end_age'] )
        del f['properties']['end_age']
    else:
        end.text = str( 200 )

    rot_id = feature_root.find('./gpml:reconstructionPlateId/gpml:ConstantValue/gpml:value', namespaces=ns)
    if 'rot_id' in f['properties'] :
        rot_id.text = str( f['properties']['rot_id'] )
        del f['properties']['rot_id']
    else: 
        rot_id.text = str( 0 )
        
    # get the feature type 
    f_type = f['type']

    # remove that prop
    #del( f['properties']['feature_type'] )

    # Create any additional properties 
    for prop in f['properties'] :
        value = f['properties'][prop]

        # print( now(), 'create_gpml_feature_collection_from_list_of_features: property =', prop, '; value = ', value)

        if prop == 'unclassifiedGeometryCoverage' :
            feature_root.append( create_gpml_unclassifiedGeometryCoverage(value) )

        # FIXME: other specialized props can go here

        else: # it's a nice normal tag , just add it as a new element 
            prop_tag = et.SubElement(feature_root, 'gpml:' + prop)
            prop_tag.text = str( value )

    # create a new feature member element for the working feature 
    feature_member = et.Element('gml:featureMember')

    # make a new feature element with the correct type 
    new_feature = et.Element(f_type)
 
    # copy to new type 
    for child in list( feature_root ):
        #print( now(), 'create_gpml_feature_collection_from_list_of_features: tag = ', child.tag)
        new_feature.append(child)

    # append the feature to the new gml:featureMember element 
    feature_member.append( new_feature )

    # append the feature to the new gml:featureMember element 
    #feature_member.append( feature_root )

    # return the element tree for this feature 
    return feature_member

#=====================================================================
#=====================================================================
def create_gpml_unclassifiedGeometryCoverage(values):
    '''create the element tree for gpml:unclassifiedGeometryCoverage'''

    #print(now(), 'create_gpml_unclassifiedGeometryCoverage: values = ', values)

    # read the gpml block
    
    # create a new boundary element from geodynamic_framework_data file
    gpml_file = os.path.dirname(__file__) + '/geodynamic_framework_data/' + 'gpml_unclassifiedGeometryCoverage.gpml'
    gpml_tree = et.parse( gpml_file )
    gpml_root = gpml_tree.getroot()
    coverage = gpml_root
    #print(now(), 'create_gpml_unclassifiedGeometryCoverage: coverage.tag = ', coverage.tag)

    # set the tuple list
    e = coverage.find('./gpml:ConstantValue/gpml:value/gml:DataBlock/gml:tupleList', namespaces=ns)

    string = ' '.join( str(x) for x in values)
    e.text = string
    
    #print(now(), 'create_gpml_unclassifiedGeometryCoverage: e.tag = ', e.tag)
    #print(now(), 'create_gpml_unclassifiedGeometryCoverage: e.txt = ', e.text)

    return coverage

#=====================================================================
#=====================================================================
def create_data_for_displacement_workflow():
    '''a place holder to help develop displacement workflows'''

    # list of features to create 
    
    # general point feature 
    feature = { 
            'properties' : 
              {
                 'rot_id': 0, 
                 'name' : 'point',
                 'description' : 'displacement point: ',
                 'begin_age' : 100,
                 'end_age' : 0,
                 'geometry_type' : 'point',
                 'geometry' : []
              },
             'displacements' : [],
          }

    # create a set of point features for the opening of a basin:
    # a system of two comensurate meshes, left and right 
    system = []
    west_mesh = { 'file_name_prefix': 'west_mesh', 'boundary' : [], 'interior' : [] }  
    east_mesh = { 'file_name_prefix': 'east_mesh', 'boundary' : [], 'interior' : [] }  

    # set up a default position
    lat0 = 0.00001
    lon0 = 0.00001

    # set up working lists
    west_tmp_list = []
    west_mesh['interior'] = []

    # loop over values in Lat and populate the outer boundaries 
    for i,y in enumerate( range(5) ) :

        print('i=', i)

        # create features for the western side of the west mesh 
        f = copy.deepcopy( feature )
        f['properties']['name'] = 'west_mesh_westside_boundary_point_' + str(i)
        f['properties']['description'] = 'displacement point: ' + f['properties']['name']
        f['properties']['geometry'].append( { 'lon' : -15, 'lat' : lat0 + y } )
        west_mesh['boundary'].append( copy.deepcopy( f ) )

        # create features for the eastern side of the west mesh 
        f = copy.deepcopy( feature )
        f['properties']['name'] = 'west_mesh_eastside_boundary_point_' + str(i)
        f['properties']['description'] = 'displacement point: ' + f['properties']['name']
        f['properties']['geometry'].append( { 'lon' : -5, 'lat' : lat0 + y } )
        west_tmp_list.append( copy.deepcopy( f ) )

        # SAVE  
        # create features for the interior of the west mesh
        #f = copy.deepcopy( feature )
        #f['properties']['name'] = 'west_interior_point_' + str(i)
        #f['properties']['description'] = 'displacement point: ' + f['properties']['name']
        #f['properties']['geometry'].append( { 'lon' : -10, 'lat' : lat0 + y } )
        #west_mesh['interior'].append( copy.deepcopy( f ) )

    # end of loop 

    # SAVE : test a single interior point 
    p = copy.deepcopy( feature)
    p['properties']['name'] = 'west_interior_point_0' 
    p['properties']['description'] = 'displacement point: ' + f['properties']['name']
    p['properties']['geometry'].append( { 'lon' : -10, 'lat' : 0.0001 } )
    west_mesh['interior'].append( copy.deepcopy( p ) )


    # establish a base rotation id number for the moving boundary 
    rot_id_base = 1011111
    # loop over boundary features and add displacements
    for i,f in enumerate( west_tmp_list ) :

        if verbose : print(Core_Util.now(), 'processing feature #', i, ' name = ', f['properties']['name'] )

        # update rotation id
        f['properties']['rot_id'] = rot_id_base + i 

        # get initial present day position 
        lat0 = f['properties']['geometry'][0]['lat']
        lon0 = f['properties']['geometry'][0]['lon']

        displacements = f['displacements']

        # SAVE : set D to non-zero to have the boundary displace 
        D = 0
        # create displacement steps, 0Ma, 1Ma, 2Ma ... DMa
        for i in range(D):
          displacements.append( 
            {
             'reconstruction_age' : i, # in Ma
             'position' : { 'lat' : lat0, 'lon' : lon0 + i },
             #'position' : { 'lat' : lat0 + i, 'lon' : lon0 },
            } 
         )
    # end of loop over boundary 

    # loop over interior features and add displacements 
    rot_id_base = 1012222
    for i,f in enumerate( west_mesh['interior'] ) :

        if verbose : print(Core_Util.now(), 'processing feature #', i, ' name = ', f['properties']['name'] )

        # update rotation id
        f['properties']['rot_id'] = rot_id_base + i 

        # get initial present day position 
        lat0 = f['properties']['geometry'][0]['lat']
        lon0 = f['properties']['geometry'][0]['lon']

        displacements = f['displacements']

        # create displacement steps, 0Ma, 1Ma, 2Ma ... DMa
        D = 10
        for i in range(D):
          displacements.append( 
            {
             'reconstruction_age' : i, # in Ma
             #'position' : { 'lat' : lat0 + i/2, 'lon' : lon0 }, # NS only motion 
             #'position' : { 'lat' : lat0, 'lon' : lon0 + i/2 }, # EW only motion 
             'position' : { 'lat' : lat0 + i/2, 'lon' : lon0 + i/2 }, # mixed motion
            } 
         )
    # end of loop over interior 

    # reverse the tmp list so the network boundary will be in proper circular order 
    west_tmp_list.reverse() 

    # append tmp to the boundary to close the loop
    west_mesh['boundary'] = west_mesh['boundary'] + west_tmp_list
    
    system.append(west_mesh)

    # NOTE: this is only one mesh, but the system could have more meshes ,,, 

    return (system) 

#=====================================================================
#=====================================================================
def test_displacement_workflow( system ) :
    ''' test of displacement workflows '''

    if verbose : print(Core_Util.now(), 'test_displacement_workflow: START')

    # create feature collections from feature displacment data 
    feature_collection = create_displacement_feature_collections( system[0] )

    # the fixed rotation id to tie all displacement rotations to 
    fixed_rot_id = 0 

    # compute Euler Pole rotations from displacements
    rotations = process_features_with_displacements_into_poles( system[0], fixed_rot_id )

    write_rotations_as_dot_rot( rotations, system[0]['file_name_prefix'] + '.rot' )

#=====================================================================
#=====================================================================
def get_base_feature():
    '''return a base feature dict'''

    # A feature is a dictionary with two keys: 'properties' and 'displacements'.
    # 'properties' is requrired for all features.
    # 'geometry' is requrired for all features.
    # 'displacements' is not required for general feature creation.
    feature = { 
        # The feature type may be changed 
        'type' : 'gpml:UnclassifiedFeature',
        # The value for 'properties' is a single dictionary:
        'properties' : {
            #'feature_type' : 'gpml:UnclassifiedFeature',
            'rot_id': 0, 
            'name' : 'test_feature',
            'description' : 'test_feature',
            'begin_age' : 100,
            'end_age' : 0,
        },
        # The value for 'geometry' is a dictionary with these keys:
        #     'type' must be one of: 'Point', 'LineString', 'Polygon'
        #     'coordinates' is a list of coordinate lists: [lat, lon]
        # Point geoms only have a single item
        # LineString geoms must have at least two lists 
        # Polygon geoms must have the same point for the first and last entry
        'geometry' : { 
            'coordinates': [], 
            'type': 'UnclassifiedGeometry'
        },
        # The value for 'displacement' is a list of dictionaries:
        #   {
        #      'reconstruction_age' : age_in_ma, 
        #      'position' : { 'lat' : float_value, 'lon' : float_value },
        #   },
        # Each dict shall have the 'position' of the feature at the 'reconstruction_age'.
        # These are absolute positions at those times (not accumulated positions)
        # The value for 'reconstruction_age' is a float or int age in Ma
        # The value for 'position' is list of coordinate pair dictionaries: 
        #   {'lat' : 10.0, 'lon' : 10.0 }, ... 
        'displacements' : [],
    }

    return feature

#=====================================================================
#=====================================================================
def json_to_gpml(gpml_file_name, json_string):
    '''convert from json string to gpml file'''

    json_d = json.loads(json_string)

    if verbose : 
        print('json_to_gpml: json_d =')
        Core_Util.tree_print( json_d )

    # Get the list of features from the JSON dict 
    features = json_d['features']

    # create a gpml file
    create_gpml_feature_collection_from_list_of_features( gpml_file_name, features)
#=====================================================================
#=====================================================================
#=====================================================================
def txt_to_gpml(file_name):
    '''Read a txt file with tab sparated columns and create gpml features in a file'''

    # empty list of features to create 
    features = []

    # read the input data file 
    if verbose: print( 'txt_to_gpml: open:', file_name )
    file = open( file_name )
    try:
        lines = [ line.rstrip() for line in file if line[0] != '#' ]
    finally:
        file.close()

    # parse lines; Skip header
    for l in lines[1:] :
    
        if verbose: print( 'txt_to_gpml: line:', l )

        # parse the line
        # FIXME: find a way to make this read the header line 
        (name, lat, lon) = l.split('\t')

        # clean up name string
        name = name.replace('"', '')
        
        # get the general feature dict 
        f = copy.deepcopy( get_base_feature() )

        # set up properties 
        f_type = 'gpml:UnclassifiedFeature'
        begin_age = 100
        end_age = 0

        # set geometry data
        g_type = 'Point'
        coords = [lat, lon]

        # create the feature dict
        f['properties']['name'] = name 
        f['properties']['begin_age'] = begin_age 
        f['properties']['end_age'] = end_age 

        f['geometry']['type'] = g_type
        f['geometry']['coordinates'] = coords

        # Append the feature to the list
        features.append( f ) 

    # end of list over lines

    # show one of the features
    if verbose : Core_Util.tree_print( features[-1] )

    # create the feature collection
    new_file_name = file_name.replace('.txt', '.gpml')

    # write out the gpml file 
    create_gpml_feature_collection_from_list_of_features( new_file_name, features )
    
#=====================================================================
#=====================================================================
#=====================================================================
def create_deformation_feature():
    '''create a multipoint feature for crustal thickness workflow'''

    print(now(), 'create_deformation_feature:')
    # parse the input .cfg file
    # read the second command line argument as a .cfg file 
    cfg_d = Core_Util.parse_configuration_file( sys.argv[2] )
    print(now(), 'create_deformation_feature: input cfg=')
    Core_Util.tree_print( cfg_d ) 

    # A list of features to create ; 
    # NOTE: only creating one feature, but creation function takes a list
    #features = []

    # get the general feature dict 
    f = copy.deepcopy( get_base_feature() )

    # set up properties 
    f_type = 'gpml:UnclassifiedFeature'
    begin_age = cfg_d['begin']
    end_age = cfg_d['end']
    name = cfg_d['name']

    # create the feature dict
    f['type'] = f_type 
    f['properties']['name'] = name 
    f['properties']['begin_age'] = begin_age 
    f['properties']['end_age'] = end_age 

    # Get geometry data
    g_type = 'unclassified_geom_multipoint'
    coord_l = []

    # Scatter points in bounding box
    lat_l = np.random.uniform( cfg_d['min_lat'], cfg_d['max_lat'], cfg_d['number_points'] )
    lon_l = np.random.uniform( cfg_d['min_lon'], cfg_d['max_lon'], cfg_d['number_points'] )
    #print(now(), 'create_deformation_feature: lat_l = ', lat_l)
    #print(now(), 'create_deformation_feature: lon_l = ', lon_l)

    # set some filenames
    tmp1_filename = 'tmp_all_points.xy'
    tmp2_filename = 'tmp_clipped_points.xy'
    tmp3_filename = 'tmp_tracked_values.txt'

    # create an .xy file with these points 
    tmp1=np.column_stack((lon_l,lat_l))
    np.savetxt(tmp1_filename, tmp1, fmt='%10.4f')

    # Select points inside the boundary
    Core_GMT.callgmt( 'gmtselect', \
       tmp1_filename, \
       {'F':cfg_d['clip_boundary'], 'fg':''}, \
       '>', \
       tmp2_filename ) 

    # read the clipped points
    lon_l, lat_l = np.loadtxt( tmp2_filename, usecols=[0, 1], unpack=True )

    #print(now(), 'create_deformation_feature: lat_l = ', lat_l)
    #print(now(), 'create_deformation_feature: lon_l = ', lon_l)

    # append the clipped coords to the list
    for i, lon in enumerate( lon_l ) :
        lat = lat_l[i]
        coord_l.append( [lon, lat] )

    # Set the Coverage values 
    values = []

    if 'initial_values_grid' in cfg_d: 

        # set the Coverage values from a grid
        grid = cfg_d['initial_values_grid']

        # Create a grid of initial crustal thickness values from the age grid
        tmp_filename4 = 'tmp_crustal_thickness_grid.grd'

        tc = cfg_d['initial_thickness_continent']
        to = cfg_d['initial_thickness_ocean']
        td = tc - to

        # grdmat %(afile_out1)s -0.5 LT %(ThkDiff)f MUL %(ThkO)f ADD = %(afile_out1)s
        arg = '%(grid)s -0.5 LT %(td)f MUL %(to)f ADD' % vars()
        Core_GMT.callgmt( 'grdmath', arg, {}, '=', tmp_filename4 ) 
        
        # read the grid and sample the values at clipped locations 
        Core_GMT.callgmt( 'grdtrack', tmp2_filename, {'G':tmp_filename4, 'Q':'l'}, '>', tmp3_filename ) 

        # Set the values from the track
        values = np.loadtxt( tmp3_filename, usecols=[2], unpack=True )

        # clean up
        Core_Util.remove_files( [tmp1_filename, tmp2_filename, tmp3_filename, tmp_filename4] )
        
    elif 'initial_values_list' in cfg_d :

        # set the Coverage values from an initial list in the .cfg file
        values = cfg_d['initial_values_list']

    else :

        # set the Coverage values from random values 
        values = np.random.uniform( 0, 10 , len( lon_l ) )

    # Set the Coverage valuesp property
    f['properties']['unclassifiedGeometryCoverage'] = values

    # set the geom
    f['geometry']['type'] = g_type
    f['geometry']['coordinates'] = coord_l

    # Append the feature to the list
    #features.append( f ) 

    # create the feature collection
    new_file_name = cfg_d['output']

    # write out the gpml file 
    # NOTE: creation function takes a list of features 
    create_gpml_feature_collection_from_list_of_features( new_file_name, [f]  )
    
#=====================================================================
#=====================================================================
def test_feature_creation_workflow():
    ''' set up some sample data and write out gpml'''

    # A feature is a dictionary with two keys: 'properties' and 'displacements'.
    # 'properties' is requrired for all features.
    # 'geometry' is requrired for all features.
    # 'displacements' is not required for general feature creation.
    feature = { 
        # The feature type may be changed 
        'type' : 'gpml:UnclassifiedFeature',
        # The value for 'properties' is a single dictionary:
        'properties' : {
            #'feature_type' : 'gpml:UnclassifiedFeature',
            'rot_id': 0, 
            'name' : 'test_feature',
            'description' : 'test_feature',
            'begin_age' : 100,
            'end_age' : 0,
        },
        # The value for 'geometry' is a dictionary with these keys:
        #     'type' must be one of: 'Point', 'LineString', 'Polygon'
        #     'coordinates' is a list of coordinate lists: [lon, lat]
        # Point geoms only have a single item
        # LineString geoms must have at least two lists 
        # Polygon geoms must have the same point for the first and last entry
        'geometry' : { 
            'coordinates': [], 
            'type': 'UnclassifiedGeometry'
        },
        # The value for 'displacement' is a list of dictionaries:
        #   {
        #      'reconstruction_age' : age_in_ma, 
        #      'position' : { 'lat' : float_value, 'lon' : float_value },
        #   },
        # Each dict shall have the 'position' of the feature at the 'reconstruction_age'.
        # These are absolute positions at those times (not accumulated positions)
        # The value for 'reconstruction_age' is a float or int age in Ma
        # The value for 'position' is list of coordinate pair dictionaries: 
        #   {'lat' : 10.0, 'lon' : 10.0 }, ... 
        'displacements' : [],
    }

    # set up a list of features
    features = [] 

    # start up the json feature collection string
    json_fc = '''{"type": "FeatureCollection", "features": ['''

    #
    # Create some random points 
    #
    for i in range(3) :

        # get the general feature dict 
        f = copy.deepcopy( feature )

        # set up properties 
        f_type = 'gpml:UnclassifiedFeature'
        name = f['properties']['name'] + str(i)
        begin_age = 100
        end_age = 0

        # set geometry data
        g_type = 'Point'
        # get a random point  
        lat = float( np.random.random_integers(-70, 70) )
        lon = float( np.random.random_integers(-170, 170) )
        coords = [lon, lat]

        # create the feature dict
        f['properties']['name'] = name 
        f['properties']['begin_age'] = begin_age 
        f['properties']['end_age'] = end_age 

        f['geometry']['type'] = g_type
        f['geometry']['coordinates'] = coords

        # Append the feature to the list
        features.append( f ) 

        # create the feature json string 
        jf = '''{"geometry": {"type": "%(g_type)s", "coordinates": [%(lat)f, %(lon)f]}, "type": "Feature", "properties": {"name": "%(name)s", "feature_type": "%(f_type)s", "begin_age": "%(begin_age)f", "end_age": "%(end_age)f"}},''' % vars()

        # append that string to the main string 
        json_fc += jf

    #
    # Create a multi point feature with some random points 
    #
    for i in range(1) :

        # get the general feature dict 
        f = copy.deepcopy( feature )

        # set up properties 
        f_type = 'gpml:UnclassifiedFeature'
        name = f['properties']['name'] + str(i)
        begin_age = 100
        end_age = 0

        # set geometry data
        g_type = 'MultiPoint'
        coord_l = []
        # get some random points 
        for j in range(5) : 
            lat = float( np.random.random_integers(-70, 70) )
            lon = float( np.random.random_integers(-170, 170) )
            coord_l.append( [lon, lat] )

        # create the feature dict
        f['properties']['name'] = name 
        f['properties']['begin_age'] = begin_age 
        f['properties']['end_age'] = end_age 

        f['geometry']['type'] = g_type
        f['geometry']['coordinates'] = coord_l

        # Append the feature to the list
        features.append( f ) 

        # process coordinate data into json string 
        coord_s = '['
        for c in coord_l:
            coord_s += '[%f, %f],' % (c[0],c[1])
        coord_s = coord_s[0:-1] + ']'

        # create the feature json string 
        jf = '''{"geometry": {"type": "%(g_type)s", "coordinates": %(coord_s)s}, "type": "Feature", "properties": {"name": "%(name)s", "feature_type": "%(f_type)s", "begin_age": "%(begin_age)f", "end_age": "%(end_age)f"}},''' % vars()

        # append that string to the main string 
        json_fc += jf

    #
    # Create some random lines 
    #
    for i in range(0) :

        # get the general feature dict 
        f = copy.deepcopy( feature )

        # set up properties 
        f_type = 'gpml:UnclassifiedFeature'
        name = f['properties']['name'] + str(i)
        begin_age = 100
        end_age = 0

        # set geometry data
        g_type = 'LineString'
        # get a random point  
        lat = np.random.random_integers(-70, 70)
        lon = np.random.random_integers(-170, 170)
        # Create a two point line 
        p1 = [lat + 0.0, lon + 0.0]
        p2 = [lat + 2.0, lon + 2.0]
        coord_l = [p1, p2]

        # update the feature dict 
        f['properties']['name'] = name 
        f['properties']['begin_age'] = begin_age 
        f['properties']['end_age'] = end_age 

        f['geometry']['type'] = g_type
        f['geometry']['coordinates'] = coord_l

        # Append the feature to the list
        features.append( f ) 

        # process coordinate data into json string 
        coord_s = '['
        for c in coord_l:
            coord_s += '[%f, %f],' % (c[0],c[1])
        coord_s = coord_s[0:-1] + ']'

        # create the feature json string 
        jf = '''{"geometry": {"type": "%(g_type)s", "coordinates": %(coord_s)s}, "type": "Feature", "properties": {"name": "%(name)s", "feature_type": "%(f_type)s", "begin_age": "%(begin_age)f", "end_age": "%(end_age)f"}},''' % vars()

        # append that string to the main string 
        json_fc += jf

    #
    # Create some random polygons 
    #
    for i in range(0) :
        f = copy.deepcopy( feature )

        # set up properties 
        f_type = 'gpml:UnclassifiedFeature'
        name = f['properties']['name'] + str(i)
        begin_age = 100
        end_age = 0

        # get a random point  
        lat = np.random.random_integers(-70, 70)
        lon = np.random.random_integers(-170, 170)

        # Create a little polygon box around the point 
        p1 = [lat + 2.0 , lon + 2.0 ]
        p2 = [lat + 2.0 , lon - 2.0 ]
        p3 = [lat - 2.0 , lon - 2.0 ]
        p4 = [lat - 2.0 , lon + 2.0 ]
        p5 = [lat + 2.0 , lon + 2.0 ] # NOTE: GPlates requires a closed loop with duplicate first and last points 
        coord_l = [p1, p2, p3, p4, p5]

        # update the feature dict 
        f['properties']['name'] = name 
        f['properties']['begin_age'] = begin_age 
        f['properties']['end_age'] = end_age 

        f['geometry']['type'] = g_type
        f['geometry']['coordinates'] = coord_l

        # Append the feature to the list
        features.append( f ) 

        # process coordinate data into json string 
        coord_s = '['
        for c in coord_l:
            coord_s += '[%f, %f],' % (c[0],c[1])
        coord_s = coord_s[0:-1] + ']'

        # create the feature json string 
        jf = '''{"geometry": {"type": "%(g_type)s", "coordinates": %(coord_s)s}, "type": "Feature", "properties": {"name": "%(name)s", "feature_type": "%(f_type)s", "begin_age": "%(begin_age)f", "end_age": "%(end_age)f"}},''' % vars()

        # append that string to the main string 
        json_fc += jf

    # show one of the features
    if verbose : Core_Util.tree_print( features[-1] )

    # create the feature collection
    file_name = './TEST_from_dict.gpml'
    # write out the gpml file 
    create_gpml_feature_collection_from_list_of_features( file_name, features )

    #remove final ',' from looping over features above 
    json_fc = json_fc[0:-1]
    # close the json feature collection
    json_fc += ''']}'''

    # capture the full JSON string as a file:
    out_file = open('./TEST.json', 'w')
    out_file.write( json_fc )
    out_file.close()
    #if verbose : Core_Util.tree_print( json_fc )

    # convert the json to gpml and write
    file_name = './TEST_from_json.gpml'
    # create the gpml file from the json string 
    json_to_gpml( file_name, json_fc )

#=====================================================================
#=====================================================================
def test( argv ):
    '''geodynamic framework module self test'''
    global verbose
    verbose = True 

    if verbose: print(now(), 'test: sys.argv = ', sys.argv )

    # run the tests 

    # read the first command line argument as a .cfg file 
    # cfg_d = parse_configuration_file( sys.argv[1] )

    # test read of gpml file 
    #transform_gpml_file_network_to_polygon( sys.argv[1] )

    # test displacement_to_euler_pole
    #displacement_to_euler_pole(0.0, 0.0, 1.0, 0.0)
    #sys.exit(0)

    # test the displacement workflow 
    #system = create_data_for_displacement_workflow()
    #if verbose : print(Core_Util.now(), 'system = ')
    #if verbose : Core_Util.tree_print( system )
    #test_displacement_workflow( system )

    # Test for feature creation 
    #test_feature_creation_workflow()

#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this module'''

    text = '''#=====================================================================
# example config.cfg file for the Core_GPlates.py
# This is a cfg for the Deformation Workflow

# Set the basic data for the feature 
name=Deformation_test_feature
plate_id=101
begin=200 ; # age of appearance 
end=0 ; # age of disapparance

# number of points in the multipoint 
number_points=1000

# bounding box of region for point generation
min_lat=14.0
max_lat=50.0
min_lon=-130.0
max_lon=-95.0

# Topology Network polygon to clip points
clip_boundary = /home/mturner/work/repos/gplates/utils/dev/sample_data/sample_data_from_gplates/topology_network_polygons_0.00Ma.xy

# NOTE: choose one of these options:

# Option 1: Set the initial scalar coverage values from an age grid 
# and initial values for ocean and continenent:
initial_values_grid=/net/beno/raid2/nflament/Agegrids/20141024_svn288/WithContinents/agegrid_final_with_continents_0.grd
initial_thickness_continent=40
initial_thickness_ocean=0

# Option2: Set the initial scalar coverage values in a list
# the list must have number_points entries 
# initial_values=10,10,10,10 ... 

# final output file name 
output=deformation_scalar_coverage_test.gpml
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
if __name__ == "__main__":
    import Core_GPlates

    if len( sys.argv ) > 1:

        # make the example configuration file 
        if sys.argv[1] == '-e':
            make_example_config_file()
            sys.exit(0)

        # batch mode to process a cmd like: Core_GPlates -net2polygon *.gpml 
        if sys.argv[1] == '-net2polygon':
            batch_transform_net_to_polygon()
            sys.exit(0)

        # batch mode to process a cmd like: Core_GPlates -json2gpml *.json 
        if sys.argv[1] == '-json2gpml':
            batch_transform_json_to_gpml()
            sys.exit(0)

        # batch mode to process a cmd like: Core_GPlates -iodpkml2gpml *.kml 
        if sys.argv[1] == '-iodpkml2gpml':
            batch_transform_iodpkml_to_gpml()
            sys.exit(0)

        # batch mode to process a cmd like: Core_GPlates -txt2gpml *.txt 
        if sys.argv[1] == '-txt2gpml':
            batch_transform_txt_to_gpml()
            sys.exit(0)

        # create the deformation feature 
        if sys.argv[1] == '-def':
            create_deformation_feature()
            sys.exit(0)

        # process sys.arv as file names for testing 
        if sys.argv[1] == '-t':
            test( sys.argv )
            sys.exit(0)
    else:
        # print module documentation and exit
        help(Core_GPlates)
#=====================================================================
#=====================================================================
