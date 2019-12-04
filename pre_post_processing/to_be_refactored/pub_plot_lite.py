#!/usr/bin/env python
#=====================================================================
#
#               Python Scripts for CitcomS Version 3.0
#                  ---------------------------------
#
#                              Author:
#                           Dan J. Bower
#          (c) California Institute of Technology 2013
#
#               Free for non-commercial academic use ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright July 2011, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
#
#=====================================================================
#
# Last Update: Dan J. Bower, 29th March 2013
#=====================================================================

import sys, string, os, re
from Core_GMT import callgmt
from Core_Util import now
from subprocess import Popen, PIPE
import numpy as np
import Core_Util, Core_Citcom, Core_GMT

# global variables
#====================================================================
gmt_base_opts
verbose = True
#====================================================================
#====================================================================
#====================================================================
def usage():
    """print usage message and exit"""

    print('''pub_plot_lite.py plot_cfg_file''')

    sys.exit()

#====================================================================
#====================================================================
#====================================================================
def main():
    """main sequence of script actions"""

    print( now(), 'pub_global.py:' )
    print( now(), 'main:' )

    # parse cmd line input for input plotting config file
    if len(sys.argv) != 2:
        usage()

    # this part is not intuitive to an uniformed user
    # can we avoid these initialize commands?
    # initialize the modules
    Core_Util.initialize()
    Core_Citcom.initialize()

    # Get the framework dictionary
    geoframe_dict = Core_Util.geoframe_dict

    # read settings from control file
    dict = Core_Citcom.parse_configuration_file( sys.argv[1] )

    # move loose parameters (not within Figure_X) from dict to a 
    # temporary new dictionary (adict) then deepcopy to
    # dict['All_Figure']
    # this cleans up dict by ensuring the keys are e.g.
    # 'All_Figure', 'Figure_A', 'Figure_B' etc.
    #adict = {}
    #for key in list(dict):
    #    if not key.startswith('Figure'):
    #        adict[key] = dict.pop(key)
    #dict['figure_keys'] = sorted(dict.keys())

    # ??? set_global_defaults( adict )
    #dict['All_Figure'] = copy.deepcopy( adict )
    #del adict # delete temporary dictionary

    # ??? set_positioning( dict )

    # set adict as pointer to dict['All_Figure']
    #adict = dict['All_Figure']

    print(dict)

    ps = 'test.ps'

    make_postscript( dict, ps )

#====================================================================
#====================================================================
#====================================================================
def make_postscript( dict, ps ):

    global gmt_base_opts

    # gmtset
    gmt_opts = {k.split()[1]: dict[k] for k in dict if k.startswith('gmtset')}
    cmd = ''
    for key, value in dict.items():
        cmd += '%(key)s %(value)s' % vars()
    callgmt( 'gmtset', cmd, '', '', '')

    # start postscript and build base GMT options dictionary
    # these parameters will not change for a given figure
    # except at the end for label plotting, e.g., J, R, X, Y
    # defined by single (length) options in the input dictionary
    gmt_base_opts = Core_GMT.start_postscript( ps )
    for key, value in dict.items():
        if len(key) == 1: gmt_base_opts[key] = value

    # psbasemap
    gmt_opts = get_gmt_options( 'psbasemap' )
    callgmt( 'psbasemap', '', gmt_opts, '>>', ps )
    del gmt_opts

    # grdimage
    if 'age_grdimage' in dict: age = dict['age_grdimage']
    else: age = dict.get('age',0)
    grid_file = locate_grid_file( grid, depth, age, dict )
    gmt_opts = get_gmt_options( 'grdimage' )
    callgmt( 'grdimage', grid_file, gmt_opts, '>>', ps )
    del gmt_opts

    # grdcontour (list)
    #grdcontour_list = something.split()
    #for entry in grdcontour_list:
    #    similar commands as grdimage
    #    Core_Util.get_current_or_last_list_entry()
    #    if age_grdcontour in dict: age = dict['age_grdcontour']
    #    else: age = dict.get('age',0)
    #    get_gmt_options( 'grdcontour' ) # to get from list though?
    #    {k.split('_')[1]: dict[k] for k in dict if k.startswith('psbasemap') and dict[k] is not '-'}

    # psxy (list)
    #psxy_list = something.split()
    #for entry in psxy_list:
    #    similar commands as grdimage

    # if map == True:
    #    if age_overlay in dict: age = dict['age_overlay']
    #    else: age = dict.get('age',0)
    #    overlay_plate_polygon
    #    overlay_slab_polygon
    #    overlay_coastline
    #    overlay_ridge_and_transform
    #    overlay_subduction
    #    overlay_slab_leading_edge
    #    overlay_gplates_velocity
    #    overlay_citcom_velocity
    #    overlay_great_circle


    # example stdin
    #gmt_opts = {'R':'0/8.5/0/11', 'J':'x1.0'}
    #stdin = '2 3 12 0 4 ML this is a test\n'
    #stdin += '4 6 12 0 4 ML this is a test 2\n'
    #stdin += 'EOF'
    #callgmt('pstext','',gmt_opts,'<< EOF >' % vars(),'test.ps\n%(stdin)s' % vars())
    #callgmt('echo %(stdin)s | pstext' % vars(),'',gmt_opts,'>','test.ps')

    # GMT logo and timestamp
    # if something: GMT logo and timestamp

    # end postscript
    Core_GMT.end_postscript( ps )

#====================================================================
#====================================================================
#====================================================================
def get_gmt_options( gmtcommand ):

    global gmt_base_options

    gmt_opts = {k.split('_')[1]: dict[k] for k in dict if
       k.startswith( gmtcommand )}
    gmt_opts.update( gmt_base_opts )

    return gmt_opts

#====================================================================
#====================================================================
#====================================================================
def locate_grid_file( grid, depth, age, dict ):

    '''define the path of certain types of gridded datasets (seismic
       seafloor, etc.) from the users defaults (specified in
       geodynamic_framework_defaults.txt)'''

    # need to ask everyone to keep grid file format the same
    # e.g. model.field.depth.age?
    # also need this format for building annulus etc.
    # build rest of path name from specified depth and age
    #depth = 
    #age = 

    # seafloor age
    if grid == 'age_with_mask':
        # next entries for testing - extract this information from user defaults
        path_to_grid = '/net/beno/raid2/marias/Agegrids/20111110/Mask/'
        grid_file = path_to_grid + 'agegrid_final_mask_%(age)s.grd' % vars()
    elif grid == 'age_without_mask':
        # next entries for testing - extract this information from user defaults
        path_to_grid = '/net/beno/raid2/marias/Agegrids/20111110/NoMask/'
        grid_file = path_to_grid + 'agegrid_final_nomask_%(age)s.grd' % vars()

    # seismic models
    elif grid == 'S40RTS':
        path_to_grid = '/path/to/S40RTS/from/framework/'
        grid_file = path_to_grid + 'XXX'
    elif grid == 'TXBW':
        path_to_grid = '/path/to/TXBW/from/framework/'
        grid_file = path_to_grid + 'XXX'
    elif grid == 'Simmons10':
        path_to_grid = '/path/to/Simmons10/from/framework/'
        grid_file = path_to_grid + 'XXX'

    # else unknown, so grid location must be explicitly specified
    else:
        grid_file = grid

    return grid_file

#====================================================================
#====================================================================
#====================================================================
def make_annulus_xyz( dict ):

    # XXX BROKEN - TO FIX FOR PYTHON 3.0 AND NEW WORK FLOW

    if verbose: print now(), 'make_annulus_xyz:'
    # annulus parameters
    if dict['model'].startswith(modelpfx): arg = dict['model'][:2]
    else: arg = dict['model']
    dict.update( annulus_parameters( dict )[arg] )

    outxyz = annpfx( dict ) + '.xyz'
    outfile = open(outxyz,'w')

    Core_Util_djb.get_great_circle_proj( dict )
    dict['G'] = '0.25'

    dict['L'] = dict.get('L_GMT','0/360')
    dict['xy'] = Core_GMT_djb.project( dict )
    if dict.get('A'): del dict['A']
    if dict.get('E'): del dict['E']
    del dict['G']
    del dict['L']

    depth_list = dict['depth_list']
    grid_pfx = dict['grid']

    for depth in depth_list:
        lradius = float(dict.get('radius',6371000.0))*1E-3-depth
        dict['grid'] = grid_pfx + '%(depth)s' % vars()
        # append age for mcm model grid convention
        if arg in modelpfx:
            age = dict.get('age',0)
            dict['grid'] += '.%(age)s.grd' % vars()
        else:
            dict['grid'] += '.grd'

        track_file = Core_GMT_djb.grdtrack( dict )

        # process track file
        infile = open(track_file,'r')
        lines = infile.readlines()
        infile.close()
        for line in lines:
            cols = line.split('\t')
            dist = cols[2]
            val = cols[3] # includes \n
            lineout = '%(dist)s %(lradius)s %(val)s' % vars()
            outfile.write( lineout )

    outfile.close()

    # if cross-section
    if dict.get('L_GMT') == 'w':
        # user-defined region
        if dict.get('R'): dist = float(dict['R'].split('/')[1])
        else: dist = float(dist)
        w = 0.5*dist # aka polar angular offset
        # necessary to ensure multiples are not appended to dict['J']
        if not dict.get('pao'):
            dict['pao'] = w
            dict['J'] += '/%(w)sz' % vars()

    return outxyz

#====================================================================
#====================================================================
#====================================================================
def make_annulus_grid( dict ):

    # XXX BROKEN - TO FIX FOR PYTHON 3.0 AND NEW WORK FLOW

    if verbose: print now(), 'make_annulus_grid:'

    # output grid name
    dict['G'] = annpfx( dict ) + '.grd'

    # always make grid with maximum extent
    if dict.get('R'): dict['R2'] = dict['R']

    #if not dict.get('R'):
    Core_GMT_djb.get_high_precision_region( dict )

    # update with correct grid_min and grid_max for this field
    dict.update( annulus_grid_range( dict )[dict['field']] )
    dict.setdefault('grid_tension',0.25) # default

    # grid_increment for blockmedian already set
    dict['xyz'] = Core_GMT_djb.blockmedian( dict )

    # update with plotting grid increment
    dict['grid_increment'] = dict['grid_increment2']
    grid = Core_GMT_djb.surface( dict )

    # xyz2grd commented out
    #dict['grid_increment'] = '6/150'
    #del dict['A']
    #dict['grid'] = Core_GMT_djb.xyz2grd( dict )
    # sometimes get_great_circle_proj sets 'A'
    del dict['grid_increment']
    del dict['grid_tension']
    del dict['G']
    del dict['xyz']

    # replace user specified region for plotting
    if dict.get('R2'): dict['R'] = dict['R2']

    return grid

#====================================================================
#====================================================================
#====================================================================
def plot_psscale_and_label( dict ):

    if dict.get('psscale_B'):
        dict['B'] = dict['psscale_B']
    else: dict['L'] = ' '

    if dict.get('psscale_S')=='True':
        dict['S'] = ' '

    dict['D'] = dict.get('psscale_D')
    dict['X'] = dict['psscale_X']
    dict['Y'] = dict['psscale_Y']
    print 'psscale_X:', dict['X'],'psscale_Y:', dict['Y']
    # allow psscale_cpt to override grdimage_cpt
    if dict.get('psscale_cpt'): dict['C'] = dict['psscale_cpt']
    else: dict['C'] = dict['grdimage_cpt']
    Core_GMT_djb.psscale( dict )
    if dict.get('B'): del dict['B']
    if dict.get('L'): del dict['L']
    if dict.get('S'): del dict['S']
    del dict['C']
    del dict['D']
    if dict.get('psscale_label'):
        dict['text'] = dict['psscale_label_pfx']+dict['psscale_label']
        dict['X'] = dict['pslabel_X']
        dict['Y'] = dict['pslabel_Y']
        Core_GMT_djb.pstext( dict )

#====================================================================
#====================================================================
#====================================================================

if __name__ == "__main__":

    main()

#====================================================================
#====================================================================
#====================================================================
