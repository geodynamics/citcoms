#!/usr/bin/env python3
#
#=====================================================================
#
#               Python Scripts for CitcomS Data Assimilation
#                  ---------------------------------
#
#                              Authors:
#                    Dan J. Bower, Michael Gurnis
#          (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#
#
#=====================================================================
#
#  Copyright 2006-2015, by the California Institute of Technology.
#
#  Last update: 16th January 2015 by DJB
#=====================================================================

import bisect, datetime, operator, os, random, subprocess, sys, time
import Core_Citcom, Core_GMT, Core_Util
import math
import numpy as np
from Core_Util import make_dir, now
from Core_GMT import callgmt
from scipy import spatial
import sys, logging
sys.path.append('/opt/GMT-4.5.14/bin:GMTHOME')
verbose = False
#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    '''print usage message and exit'''

    print('''make_history_for_age.py control_file age IC

where:

    control_file:     Control file containing numerous parameters.
                      Generate a template input by calling
                      Create_History.py without any arguments.

    age:              Age (Ma) of model mantle to create.

    IC:               Enable construction of initial condition.

''')

    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
def main():

    logging.basicConfig(level=logging.INFO, 
            format='%(asctime)s %(levelname)-8s %(message)s')

    '''Main sequence of script actions.'''

    # ----------------------------------------------------------------
    # PART I - initialize
    # ----------------------------------------------------------------

    logging.info('start make_history_for_age.py' )

    if len(sys.argv) != 4:
        usage()

    # master dictionary
    master_d = basic_setup(sys.argv[1], sys.argv[2], sys.argv[3])

    # dictionaries
    coor = master_d['coor_d']
    control_d = master_d['control_d']
    func_d = master_d['func_d']
    pid_d = master_d['pid_d']

    # parameters
    BUILD_LITHOSPHERE = control_d['BUILD_LITHOSPHERE']
    BUILD_SLAB = control_d['BUILD_SLAB']
    CITCOM_EXPORT = control_d['CITCOM_EXPORT']
    DATA = control_d['DATA']
    DEBUG = control_d['DEBUG']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    OUTPUT_IVEL = control_d['OUTPUT_IVEL']
    OUTPUT_LITH_AGE = control_d['OUTPUT_LITH_AGE']
    OUTPUT_TEMP = control_d['OUTPUT_TEMP']
    OUTPUT_TEMP_IC = control_d['OUTPUT_TEMP_IC']
    OUTPUT_TRAC_IC = control_d['OUTPUT_TRAC_IC']
    SLAB_STENCIL = control_d['SLAB_STENCIL']
    SYNTHETIC = control_d['SYNTHETIC']
    UTBL_AGE_GRID = control_d['UTBL_AGE_GRID']
    rm_list = func_d['rm_list']

    # ----------------------------------------------------------------
    # PART II - get data files required for building temperature
    # ----------------------------------------------------------------
    
    if DATA:
        control_d['sub_file'] = preprocess_gplates_line_data( master_d, control_d['sub_file'] )
        if control_d['FLAT_SLAB']:
            control_d['flat_slab_leading_file'] = preprocess_gplates_line_data( master_d,
                control_d['flat_slab_leading_file'] )
            
        # afile_1 is basically always required.  Even if not building
        # slabs it is used as the GMT grid to construct uniform
        # backgrounds.  Therefore, always make afile_1.
        make_age_grid_to_build_mantle_and_slab_temp( master_d )
        # build slabs for temp ic or temp hist
        if BUILD_SLAB and (OUTPUT_TEMP_IC or OUTPUT_TEMP) or OUTPUT_IVEL:
            get_global_data( master_d )

    else: # if SYNTHETIC
        get_synthetic_data( master_d )

    # ----------------------------------------------------------------
    # PART III - build temperature grids, map to nodes
    # ----------------------------------------------------------------

    
    # write surface (lon, lat) coord files by cap
    if OUTPUT_TEMP_IC or OUTPUT_TEMP or OUTPUT_IVEL or OUTPUT_LITH_AGE:
        write_coordinates_by_cap( master_d )
    
    if UTBL_AGE_GRID and (BUILD_LITHOSPHERE or BUILD_SLAB):
        make_age_grid_to_build_utbl( master_d )

    if OUTPUT_TEMP_IC or OUTPUT_TEMP:
        build_temperature_for_all_znodes( master_d )
        grids = func_d['master_grid_list']
        background = control_d['temperature_mantle']
        min = control_d['temperature_min']
        max = control_d['temperature_max']
        temp_grids = [grid[0] for grid in grids]
        func_d['temp_by_cap'] = track_grids_to_cap_list( master_d, 
                                    temp_grids, background, min, max )

    if OUTPUT_TEMP:
        grids = func_d['master_grid_list']
        background = control_d['stencil_background']
        min = control_d['stencil_min']
        max = control_d['stencil_max']
        sten_grids = [grid[1] for grid in grids]
        func_d['sten_by_cap'] = track_grids_to_cap_list( master_d, 
                                    sten_grids, background, min, max )

    
    # ----------------------------------------------------------------
    # PART IV: make stencils, data export, and clean up
    # ----------------------------------------------------------------

    if CITCOM_EXPORT: citcom_data_export( master_d )

    # always write parameters file
    log_dir = control_d['log_dir']
    model_name = control_d['model_name']
    make_dir( log_dir )
    log_name = control_d['log_dir'] + '/' + control_d['model_name'] + \
               '.parameters'
    Core_Util.write_dictionary_to_file( control_d, log_name )

    if not DEBUG: Core_Util.remove_files( func_d['rm_list'] )

    logging.info('make_history_for_age.py completed successfully' )

#=====================================================================
#=====================================================================
#=====================================================================
def get_depth_for_looping( control_d ):

    '''determine the depth necessary for looping to build slabs given 
       the user-specified parameters'''

    # this parameter is created by get_slab_data() and gives the 
    # maximum depth of a slab for the line data and parameters
    # selected by the user.  If get_slab_data has not been called
    # then the user does not want to build slabs and 
    # control_d['slab_depth_gen'] has not been defined
    slab_depth_gen = control_d.setdefault('slab_depth_gen',0)
    stencil_depth_max = control_d['stencil_depth_max']
    stencil_smooth_max = control_d['stencil_smooth_max']

    # total depth of stencil to approximately 0 contour
    # extra 75 km to catch base of stencil
    stencil_max = stencil_depth_max + stencil_smooth_max + 75

    if slab_depth_gen < stencil_max:
        slab_depth_gen = stencil_max
        text = 'WARNING: redefining slab_depth_gen ='
        if verbose: print( now(), text, stencil_max)

    # update dictionary
    control_d['slab_depth_gen'] = slab_depth_gen

#=====================================================================
#=====================================================================
#=====================================================================
def basic_setup(cfg_filename, age, IC):

    logging.info('start some basic setup...')

    '''Read parameters from input files and set defaults.'''

    # master dictionary for all settings
    master = {} 

    # read settings from control file
    control_d = Core_Util.parse_configuration_file( cfg_filename )

    set_verbose( control_d )

    master['control_d'] = control_d 

    # update with parameters from geodynamic framework
    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    master['geoframe_d'] = geoframe_d

    # add arguments from command line to dictionary
    control_d['age'] = int(age)
    control_d['IC'] = int(IC)

    # initial condition
    if not control_d['IC']:
        control_d['OUTPUT_TEMP_IC'] = False
        control_d['OUTPUT_TRAC_IC'] = False

    # 'SYNTHETIC' is only used for benchmarking and thus looks
    # confusing in the code.  Define another switch 'DATA' for
    # typical data-driven history creation
    if not control_d['SYNTHETIC']: control_d['DATA'] = True
    else: control_d['DATA'] = False

    # parse CitcomS pid file
    pid_d = Core_Util.parse_configuration_file( control_d['pid_file'] )
    master['pid_d'] = pid_d
    pid_d.update( Core_Citcom.derive_extra_citcom_parameters( pid_d ) )

    # get radial coordinates
    control_d['coord_file'] = control_d.get('coord_dir','') + '/' + pid_d['datafile']+'.coord.#'
    pid_d['coord_type'], pid_d['coord_file_in_use'] =\
            Core_Citcom.read_citcom_coor_type( pid_d, control_d['coord_file'] )
    coor_d = Core_Citcom.read_citcom_z_coor( pid_d, control_d['coord_file'] )
    master['coor_d'] = coor_d

    # dictionary to store values for passing between functions
    master['func_d'] = {}
    # remove list
    master['func_d']['rm_list'] = []

    # read job settings
    control_d['serial'] = isSerial(control_d)

    # set some global defaults
    set_global_defaults( control_d, pid_d )

    # check input parameters
    check_input_parameters( master )

    return master

#=====================================================================
#=====================================================================
#=====================================================================
def build_mantle_temperature( master ):

    '''Create a uniform random distribution of points with the mantle
       temperature and stencil background.  Also create a GMT grid
       with all values set to the mantle temperature and stencil 
       background.'''

    control_d = master['control_d']
    func_d = master['func_d']
    afile_1 = control_d['afile_1']
    BUILD_ADIABAT = control_d['BUILD_ADIABAT']
    grid_dir = control_d['grid_dir']
    grd_res = control_d['grd_res']
    rm_list = func_d['rm_list']
    spacing_bkg_pts = control_d['spacing_bkg_pts']
    stencil_background = control_d['stencil_background']
    temperature_mantle = control_d['temperature_mantle']

    # make uniform xyz for slabs
    # required for blending together slab temp / sten with ambient
    if control_d['BUILD_SLAB'] and spacing_bkg_pts > 0:
        # mantle temperature
        temp_xyz = grid_dir + '/temperature_mantle.xyz'
        rm_list.append( temp_xyz )
        Core_Util.make_uniform_background( temp_xyz, spacing_bkg_pts,
                                                  temperature_mantle )
        # stencil
        sten_xyz = grid_dir + '/stencil_mantle.xyz'
        rm_list.append( sten_xyz )
        Core_Util.make_uniform_background( sten_xyz, spacing_bkg_pts,
                                                  stencil_background )

    else:
        temp_xyz = None # temperature
        sten_xyz = None # stencil

    # mantle temperature grid
    # sample afile_1 for simplicity
    temp_grid = grid_dir + '/temperature_mantle.grd'
    rm_list.append( temp_grid )
    cmd = 'cp %(afile_1)s %(temp_grid)s' % vars()
    subprocess.call( cmd, shell=True )
    args = '%(temp_grid)s 0 MUL %(temperature_mantle)s ADD' % vars()
    callgmt( 'grdmath', args, '', '=', temp_grid )

    # ambient stencil grid
    # sample afile_1 for simplicity
    sten_grid = grid_dir + '/stencil_mantle.grd'
    rm_list.append( sten_grid )
    cmd = 'cp %(afile_1)s %(sten_grid)s' % vars()
    subprocess.call( cmd, shell=True )
    args = '%(sten_grid)s 0 MUL %(stencil_background)s ADD' % vars()
    callgmt( 'grdmath', args, '', '=', sten_grid )

    # return two arguments, each a 2-tuple
    return (temp_grid, sten_grid), (temp_xyz, sten_xyz)

#=====================================================================
#=====================================================================
#=====================================================================
def build_slab_temperature( master, kk, mantle_xyzs ):

    '''Build (and return) a GMT grid of slab temperatures for a given
       depth node (kk).'''

    logging.info('start build_slab_temperature' )

    coor_d = master['coor_d']
    control_d = master['control_d']
    func_d = master['func_d']
    BUILD_ADIABAT = control_d['BUILD_ADIABAT']
    filter_width = str(control_d['filter_width'])
    FLAT_SLAB = control_d['FLAT_SLAB']
    FLAT_SLAB_RAMP = control_d['FLAT_SLAB_RAMP']
    grd_res = str(control_d['grd_res'])
    R = 'g'
    rm_list = func_d['rm_list']
    spacing_bkg_pts = control_d['spacing_bkg_pts']
    stencil_filter_width = str(control_d['stencil_filter_width'])
    stencil_max = str(control_d['stencil_max'])
    stencil_min = str(control_d['stencil_min'])
    temperature_mantle = str(control_d['temperature_mantle'])
    temperature_min = str(control_d['temperature_min'])
    tension = str(control_d['tension'])

    # exit
    depth_km = coor_d['depth_km'][kk]
    if depth_km > control_d['slab_depth_gen']: return ( None, None )

    # advection factor
    if depth_km < control_d['UM_depth']: advection = control_d['UM_advection']
    else: advection = control_d['LM_advection']

    # slab temperature and stencil
    slab_xyzs =  Core_Util.make_slab_temperature_xyz( master, kk )

    if FLAT_SLAB and FLAT_SLAB_RAMP:
        # flat slab temperature and stencil
        flat_xyzs = \
            Core_Util.make_flat_slab_temperature_xyz( master, kk )
        for nn, slab_xyz in enumerate( slab_xyzs ):
            cmd = 'cat ' + flat_xyzs[nn] + ' >> ' + slab_xyz
            if verbose: print( now(), cmd )
            subprocess.call( cmd, shell=True )

    # merge data with background and make grids
    slab_grids = []
    grd_mins = ( temperature_min, stencil_min )
    grd_maxs = ( temperature_mantle, stencil_max )
    filters = ( filter_width, stencil_filter_width )
    for nn, slab_xyz in enumerate( slab_xyzs ):
        cmd = 'cat ' + mantle_xyzs[nn] + ' >> ' + slab_xyz
        if verbose: print( now(), cmd )
        subprocess.call( cmd, shell=True )

        # make grid
        grd_min = grd_mins[nn]
        grd_max = grd_maxs[nn]
        filter = filters[nn]
        median_xyz = slab_xyz.rstrip('xyz') + 'median.xyz'
        rm_list.append( median_xyz )
        grid = slab_xyz.rstrip('xyz') + 'grd'
        rm_list.append( grid )
        slab_grids.append( grid )
        cmd = slab_xyz + ' -I' + grd_res + ' -R' + R
        callgmt( 'blockmedian', cmd, '', '>', median_xyz )
        cmd = median_xyz + ' -I' + grd_res + ' -R' + R + ' -T' + tension \
              + ' -Ll' + grd_min + ' -Lu' + grd_max
        callgmt( 'surface', cmd, '', '', '-G' + grid )
        # force pixel registration (this cannot be done using surface)
        args = '%(grid)s -T' % vars()
        callgmt( 'grdsample', args, '', '', '-G%(grid)s' % vars() )

        # filter grid
        cmd = grid + ' -D2 -Fg' + filter + ' -R' + R
        callgmt( 'grdfilter', cmd, '', '', '-G' + grid )

        # clip grid
        cmd = grid + ' -Sa%(grd_max)s/%(grd_max)s -Sb%(grd_min)s/%(grd_min)s' % vars()
        callgmt( 'grdclip', cmd, '', '', '-G' + grid )

    return tuple( slab_grids )

#=====================================================================
#=====================================================================
#=====================================================================
def build_temperature_for_all_znodes( master ):

    '''Main loop over znodes (depth) to build GMT temperature grids.'''
    
    logging.info('start build_temperature_for_all_znodes')
    
    control_d = master['control_d']
    coor_d = master['coor_d']
    func_d = master['func_d']
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    KEEP_PS = control_d['KEEP_PS']
    pid_d = master['pid_d']
    master_grid_list = []
    ps_list = []
    rm_list = func_d['rm_list']
    slab_grid_list = []
    spacing_bkg_pts = control_d['spacing_bkg_pts']

    # always mantle temperature and background stencil
    mantle_grids, mantle_xyzs = build_mantle_temperature( master )

    get_depth_for_looping( control_d )

    # master node points [[lonDeg,latDeg,lonRad,colatRad,x,y,z], ...]
    # are computed only once for blob/silo ICs
    master_node_points = None
    for kk in range( pid_d['nodez'] ):

        depth_km = int(coor_d['depth_km'][kk])
        suffix = '.%(depth_km)skm.' % vars() +str(control_d['age'])+'Ma.'
        control_d['suffix'] = suffix

        if verbose: print( '########################' )
        if verbose: print( now(), 'Depth=', str(depth_km), 'km' )
        if verbose: print( '########################' )

        # master grid for combined data
        temp_grid = control_d['grid_dir'] + '/temp' + suffix + 'grd'
        sten_grid = control_d['grid_dir'] + '/sten' + suffix + 'grd'
        master_grids = ( temp_grid, sten_grid )
        master_grid_list.append( master_grids )

        # copy to master grids
        for nn, grid in enumerate( mantle_grids ):
            cmd = 'cp ' + grid + ' ' + master_grids[nn]
            if verbose: print( now(), cmd )
            subprocess.call( cmd, shell=True )

        # slab temperature
        if control_d['BUILD_SLAB']:
            slab_grids = build_slab_temperature( master, kk, mantle_xyzs )
            # replace master grids
            if slab_grids[0] is not None:
                for nn, slab_grid in enumerate( slab_grids ):
                    cmd = 'cp ' + slab_grid + ' ' + master_grids[nn]
                    if verbose: print( now(), cmd )
                    subprocess.call( cmd, shell=True )
        else:
            slab_grids = ( None, None )
        slab_grid_list.append( slab_grids )

        # lithosphere temperature (always include for slab assimilation)
        if control_d['BUILD_LITHOSPHERE'] or control_d['BUILD_SLAB']:
            temp_grid, lith_grid = include_lithosphere( master, 
                                         temp_grid, kk )
        else: lith_grid = None

        # lower thermal boundary layer
        if control_d['BUILD_LTBL']:
            temp_grid = include_ltbl( master, temp_grid, kk )
            # placeholder - could add further assimilation commands
            # here to assimilate plumes at the CMB.

        # thermal blobs
        if control_d['BUILD_BLOB']:
            temp_grid, sten_grid, master_node_points = include_blob( master, master_grids, \
                                                          master_node_points, kk )
        
        # thermal silos
        if control_d['BUILD_SILO']:
            temp_grid, sten_grid, master_node_points = include_silo( master, master_grids, \
                                                              master_node_points, \
                                                              kk )

        # adiabat
        if control_d['BUILD_ADIABAT']:
            temp_grid = include_adiabat(
                              master, temp_grid, kk )

        if control_d['PLOT_SUMMARY_POSTSCRIPT']:
            ps = make_summary_postscript( master, master_grids, kk,
                mantle_grids[0], slab_grids[0], lith_grid )
            ps_list.append( ps )
        else:
            ps_list = None


    # End of z node loop

    # make summary pdf from postscript files
    if control_d['PLOT_SUMMARY_POSTSCRIPT'] and ps_list is not None:
        age = control_d['age']
        filename = control_d['ps_dir'] + '/'
        filename += '%(age)03dMa.summary.pdf' % vars()
        Core_Util.make_pdf_from_ps_list( ps_list, filename )
        # remove temperature grids by default
        if not KEEP_PS: rm_list.append( ps_list )

    # remove temperature grids by default
    if not KEEP_GRIDS: rm_list.append( master_grid_list )

    # pass values using function dictionary
    func_d['master_grid_list'] = master_grid_list
    func_d['slab_grid_list'] = slab_grid_list

#=====================================================================
#=====================================================================
#=====================================================================
def check_input_parameters( master_d ):

    '''Basic checks to ensure that necessary parameters are set
       correctly.'''

    logging.info('start checking input parameters...' )

    # dictionaries
    control_d = master_d['control_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']
    pid_d = master_d['pid_d']

    # parameters
    age = control_d['age']
    BUILD_LTBL = control_d['BUILD_LTBL']
    CONTINENTAL_TYPES = control_d['CONTINENTAL_TYPES']
    DATA = control_d['DATA']
    FLAT_SLAB = control_d['FLAT_SLAB']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    grd_res = str(control_d['grd_res'])
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    KEEP_PS = control_d['KEEP_PS']
    OUTPUT_LITH_AGE = control_d['OUTPUT_LITH_AGE']
    OUTPUT_TEMP = control_d['OUTPUT_TEMP']
    OUTPUT_TEMP_IC = control_d['OUTPUT_TEMP_IC']
    OUTPUT_IVEL = control_d['OUTPUT_IVEL']
    OUTPUT_TRAC_IC = control_d['OUTPUT_TRAC_IC']
    rm_list = func_d['rm_list']
    SERIAL = control_d['serial']
    stencil_depth_max = control_d['stencil_depth_max']
    stencil_smooth_max = control_d['stencil_smooth_max']
    SYNTHETIC = control_d['SYNTHETIC']
    temperature_cmb = control_d['temperature_cmb']
    temperature_mantle = control_d['temperature_mantle']
    velo_dir = geoframe_d['gplates_velo_grid_dir']

    # temperature and lower thermal boundary layer error handling
    if temperature_mantle > temperature_cmb:
        err = 'ERROR: temperature_mantle > temperature_cmb'
        logging.critical( err )
        exit(1)
    elif temperature_mantle == temperature_cmb and BUILD_LTBL:
        err = 'ERROR: BUILD_LTBL but temperature_mantle = temperature_cmb!'
        logging.critical( err )
        exit(1)

    # force some directories to be local otherwise files are
    # modified and removed by other instances of the code
    if not SERIAL:
        control_d['grid_dir'] = control_d['grid_dir'].split('/')[-1]

    # global or regional model
    if DATA:

        # check entries from the geodynamics framework (age grids, subduction xy)
        control_d['age1_file']  =  geoframe_d['age_grid_no_mask_dir'] + '/'
        control_d['age1_file'] += geoframe_d['age_grid_no_mask_prefix'] + '%(age)s.grd' % vars()
        control_d['sub_file']   =  geoframe_d['gplates_line_dir'] + '/'
        control_d['sub_file']  += 'topology_subduction_boundaries'
        control_d['sub_file_left'] = control_d['sub_file'] + '_sL_%0.2fMa.xy' % age
        control_d['sub_file_right'] = control_d['sub_file'] + '_sR_%0.2fMa.xy' % age
        control_d['sub_file']  += '_%0.2fMa.xy' % age

        # check for continents
        if CONTINENTAL_TYPES:
            control_d['age2_file']  =  geoframe_d['age_grid_cont_dir'] + '/'
            control_d['age2_file'] += geoframe_d['age_grid_cont_prefix'] + '%(age)s.grd' % vars()
        else:
            control_d['age2_file']  =  geoframe_d['age_grid_mask_dir'] + '/'
            control_d['age2_file'] += geoframe_d['age_grid_mask_prefix'] + '%(age)s.grd' % vars()

        # ensure that these files exist
        for ifile in [ control_d['age1_file'], control_d['age2_file'], control_d['sub_file'] ]:
            if not os.path.exists( ifile ):
                logging.critical(f'cannot find file: {ifile}' )
                sys.exit(1)

        if FLAT_SLAB:
            control_d['flat_slab_polygon_file']  =  geoframe_d['gplates_line_dir'] + '/'
            control_d['flat_slab_polygon_file'] += 'topology_slab_polygons_%0.2fMa.xy' % age
            control_d['flat_slab_leading_file']  =  geoframe_d['gplates_line_dir'] + '/'
            control_d['flat_slab_leading_file'] += 'topology_slab_edges_leading_%0.2fMa.xy' % age
            for ifile in [ control_d['flat_slab_polygon_file'], control_d['flat_slab_leading_file'] ]:
                if not os.path.exists( ifile ):
                    # turn off flat slab
                    logging.warning(f'cannot find file: {ifile}')
                    logging.warning(f'turning OFF flat slab')
                    control_d['FLAT_SLAB'] = False
                else:
                    # a start depth is required for flat slabs to work
                    # correctly, and this is in the GPML header data
                    control_d['GPML_HEADER'] = True


        # define the names of the gplates vx (colat) and vy (longitude)
        # GMT grids that are required to build the internal velocity
        # boundary conditions.
        if OUTPUT_IVEL:
            grid_names = []
            for comp in ['vx', 'vy']:
                velo_name =  velo_dir + '/'
                velo_name += 'gplates_%(comp)s.0.%(age)s.grd' % vars()
                # ensure that file exists
                if not os.path.exists( velo_name ):
                    print( now(), 'ERROR: cannot find file: %(velo_name)s' % vars() )
                    sys.exit(1)
                grid_names.append( velo_name )

            func_d['velocity_grid_names'] = grid_names

    # End DATA

    # make grid directory in preparation for data generation
    make_dir( control_d['grid_dir'] )


    #some parameters are not independent, check them and get attention from users.
    if  pid_d['Solver'] == 'multigrid':
        nodex = pid_d['mgunitx'] * int(math.pow(2,pid_d['levels']-1)) * pid_d['nprocx'] + 1
        nodey = pid_d['mgunity'] * int(math.pow(2,pid_d['levels']-1)) * pid_d['nprocy'] + 1
        nodez = pid_d['mgunitz'] * int(math.pow(2,pid_d['levels']-1)) * pid_d['nprocz'] + 1
        if nodex != pid_d['nodex'] or nodey != pid_d['nodey'] or nodez != pid_d['nodez']:
            logging.error('The equations ') 
            logging.error('nodex = 1 + nprocx + mgunitx + 2**(levels-1)')
            logging.error('nodey = 1 + nprocy + mgunity + 2**(levels-1)')
            logging.error('nodez = 1 + nprocz + mgunitz + 2**(levels-1)')
            logging.error('must be satisfied!')
            sys.exit(1)

    if 'tracers_per_element' in control_d:
        logging.error(f"the tracers_per_element parameter should be moved into {control_d['pid_file']}.")
        sys.exit(' - XXXX - preprocessing failed due to unexpected parameter in control file!')
#=====================================================================
#=====================================================================
#=====================================================================
def make_age_grid_to_build_mantle_and_slab_temp( master_d ):

    '''Process the unmasked age grid to be used to build slabs.  Also
       used to make temperature and stencil backgrounds.'''

    logging.info('start make_age_grid_to_build_mantle_and_slab_temp' )

    control_d = master_d['control_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']
    age = control_d['age']
    age1_file = control_d['age1_file']
    grd_res = control_d['grd_res']
    grid_dir = control_d['grid_dir']
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    oceanic_lith_age_max = control_d['oceanic_lith_age_max']
    lith_age_min = control_d['lith_age_min']
    R = 'g' # always global for age grid processing
    rm_list = func_d['rm_list']

    # local copy of age grid for edit
    afile_out = grid_dir + '/lith_age1_%d.grd' % age
    if not KEEP_GRIDS: rm_list.append( afile_out )
    cmd = 'cp ' + age1_file + ' ' + afile_out
    subprocess.call( cmd, shell=True )
    # save to dictionary
    control_d['afile_1'] = afile_out

    # sample to grd_res to reduce processing time
    # do not specify 'Rg' in this argument because for some
    # reason it converts lon to -360 to 0!
    logging.debug(f'sampling {afile_out} to grd_res={grd_res}' )
    args = '%(afile_out)s -I%(grd_res)s -F' % vars()
    callgmt( 'grdsample', args, '', '', '-G%(afile_out)s' % vars() )

    # truncate age between lith_age_min and oceanic_lith_age_max
    logging.debug('clipping age grid{afile_out}')
    cmd = afile_out
    cmd += ' -Sa%(oceanic_lith_age_max)s/%(oceanic_lith_age_max)s' % vars()
    cmd += ' -Sb%(lith_age_min)s/%(lith_age_min)s' % vars()
    callgmt( 'grdclip', cmd, '', '', '-G' + afile_out )

#=====================================================================
#=====================================================================
#=====================================================================
def make_age_grid_to_build_utbl( master_d ):

    '''Process the masked (or stenciled) age grid to be used to build
       the upper thermal boundary layer.'''

    logging.info('start make_age_grid_to_build_utbl' )

    control_d = master_d['control_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']
    CONTINENTAL_TYPES = control_d['CONTINENTAL_TYPES']
    age = control_d['age']
    age2_file = control_d['age2_file']
    grd_res = control_d['grd_res']
    grid_dir = control_d['grid_dir']
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    oceanic_lith_age_max = control_d['oceanic_lith_age_max']
    lith_age_min = control_d['lith_age_min']
    NaN_age = control_d['NaN_age']
    R = 'g' # always global for age grid processing
    rm_list = func_d['rm_list']

    # local copy of age grid for edit
    afile_out = grid_dir + '/lith_age2_%d.grd' % age
    if not KEEP_GRIDS: rm_list.append( afile_out )
    cmd = 'cp ' + age2_file + ' ' + afile_out
    subprocess.call( cmd, shell=True )
    # save to dictionary
    control_d['afile_2'] = afile_out

    # before considering continental regions (and/or NaNs),
    # truncate the oceanic regions to the oceanic lith max age
    # this does not affect continental regions because they are stenciled
    # with negative numbers (and NaNs also are unaffected)
    msg = 'maximum oceanic (thermal) age is %(oceanic_lith_age_max)s Ma' % vars()
    if verbose: print( now(), msg )
    if verbose: print( now(), 'clipping age grid ', afile_out )
    cmd = afile_out
    cmd += ' -Sa%(oceanic_lith_age_max)s/%(oceanic_lith_age_max)s' % vars()
    callgmt( 'grdclip', cmd, '', '', '-G' + afile_out )

    # build continents
    if CONTINENTAL_TYPES:
        if verbose: print( now(), 'building continental types' )
        stencil_ages = control_d['stencil_ages']
        stencil_values = control_d['stencil_values']
        stencil_l = list(zip( stencil_values, stencil_ages ))
        stencil_l.sort(key=lambda tup: tup[0], reverse=False)
        for (svalue, sage) in stencil_l:
            args = '%(afile_out)s %(svalue)s LE %(sage)s MUL %(afile_out)s ADD' % vars()
            callgmt( 'grdmath', args, '', '=', afile_out )

    # otherwise, NaNs in this grid can be used to define continental
    # (non-oceanic) lithosphere
    if NaN_age:
        if verbose: print( now(), 'replacing NaN' )
        cmd = '%(afile_out)s %(NaN_age)s AND ' % vars()
        callgmt( 'grdmath', cmd, '', '=', afile_out )

    # sample to grd_res to reduce processing time
    # do not specify 'Rg' in this argument because for some
    # reason it converts lon to -360 to 0!
    if verbose: print( now(), 'sampling %(afile_out)s to grd_res=%(grd_res)s' % vars() )
    args = '%(afile_out)s -I%(grd_res)s -F' % vars()
    callgmt( 'grdsample', args, '', '', '-G%(afile_out)s' % vars() )

    # determine maximum 'allowable' age of the utbl
    utbl_age_max = np.max( [oceanic_lith_age_max, NaN_age] )
    if CONTINENTAL_TYPES:
        stencil_age_max = np.max( control_d['stencil_ages'] )
        utbl_age_max = np.max( [utbl_age_max, stencil_age_max] )
    control_d['utbl_age_max'] = utbl_age_max

    if verbose: print( now(), 'clipping age grid ', afile_out )
    cmd = afile_out
    cmd += ' -Sa%(utbl_age_max)s/%(utbl_age_max)s' % vars()
    cmd += ' -Sb%(lith_age_min)s/%(lith_age_min)s' % vars()
    callgmt( 'grdclip', cmd, '', '', '-G' + afile_out )

#=====================================================================
#=====================================================================
#=====================================================================
def make_stencil_grid_to_build_tracers( master_d ):

    if verbose: print( now(), 'make_stencil_grid_to_build_tracers:' )

    # dictionaries
    control_d = master_d['control_d']
    func_d = master_d['func_d']

    # parameters
    age = control_d['age']
    age2_file = control_d['age2_file']
    grid_dir = control_d['grid_dir']
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    R = 'g' # always global for age grid processing
    rm_list = func_d['rm_list']
    SLAB_STENCIL = control_d['SLAB_STENCIL']
    TRACER_NO_ASSIM = control_d['TRACER_NO_ASSIM']

    # local copy of age grid for edit
    afile_out = grid_dir + '/lith_age4_%d.grd' % age
    if not KEEP_GRIDS: rm_list.append( afile_out )
    cmd = 'cp ' + age2_file + ' ' + afile_out
    subprocess.call( cmd, shell=True )
    # save to dictionary
    control_d['afile_4'] = afile_out # afile_4 for historical reasons

    # correct grid registration
    args = '%(afile_out)s -F' % vars()
    callgmt( 'grdsample', args, '', '', '-G%(afile_out)s' % vars() )

    if SLAB_STENCIL:
        slab_stencil_width = control_d['slab_stencil_width']
        sub_file = control_d['sub_file']
        slab_stencil_mask = grid_dir + '/slab_stencil_mask.grd'
        rm_list.append( slab_stencil_mask )
        cmd = sub_file
        # need to set -I to be consistent with input grid
        cmd += ' -R%(R)s -I0.1 -m -F' % vars()
        cmd += ' -NNaN/0/0' % vars()
        cmd += ' -S%(slab_stencil_width)fk' % vars()
        callgmt( 'grdmask', cmd, '', '', '-G' + slab_stencil_mask )
        args = '%(slab_stencil_mask)s %(afile_out)s AND' % vars()
        callgmt( 'grdmath', args, '', '=', afile_out )

    # to exclude tracers from deforming regions
    # set no assimilation regions to NaN
    if TRACER_NO_ASSIM:
        no_assimilation_regions( master_d, afile_out, 0.1 )

#=====================================================================
#=====================================================================
#=====================================================================
def no_assimilation_regions( master_d, age_grid, grd_res ):

    '''Mask no assimilation regions with NaN.'''

    if verbose: print( now(), 'building no assimilation regions' )
    if verbose: print( now(), 'working on: %(age_grid)s' % vars() )

    control_d = master_d['control_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']

    age = control_d['age']
    grid_dir = control_d['grid_dir']
    no_ass_dir = geoframe_d['no_ass_dir']
    no_ass_file = no_ass_dir + \
        '/topology_network_polygons_%(age)s.00Ma.xy' % vars() 
    padding = control_d['no_ass_padding']
    R = 'g' # by default for all age grid processing
    rm_list = func_d['rm_list']

    if os.path.exists( no_ass_file ):
        no_ass_file2 = preprocess_gplates_line_data( master_d, no_ass_file )
        # interior of polygon
        no_ass_mask1 = grid_dir + '/no_ass_mask1.grd'
        rm_list.append( no_ass_mask1 )
        cmd = no_ass_file2
        cmd += ' -R%(R)s -I%(grd_res)g -m -F' % vars()
        cmd += ' -N0/NaN/NaN' % vars()
        callgmt( 'grdmask', cmd, '', '', '-G' + no_ass_mask1 )

        if padding: 
            # extra padding around edge of polygon
            no_ass_mask2 = grid_dir + '/no_ass_mask2.grd'
            rm_list.append( no_ass_mask2 )
            cmd = no_ass_file2
            cmd += ' -R%(R)s -I%(grd_res)g -m -F' % vars()
            cmd += ' -N0/NaN/NaN' % vars()
            cmd += ' -S%(padding)fk' % vars()
            callgmt( 'grdmask', cmd, '', '', '-G' + no_ass_mask2 )
            args = '%(age_grid)s %(no_ass_mask2)s %(no_ass_mask1)s OR OR' % vars()
            callgmt( 'grdmath', args, '', '=', age_grid )
        else:
            args = '%(age_grid)s %(no_ass_mask1)s OR' % vars()
            callgmt( 'grdmath', args, '', '=', age_grid )

    else:
        # turn off no assimilation regions
        print( now(), 'WARNING: cannot find file: %(no_ass_file)s' % vars())
        print( now(), 'WARNING: aborting no assimilation regions')

#=====================================================================
#=====================================================================
#=====================================================================
def track_grids_to_cap_list( master, grid_list, background, min, max ):

    '''Export gridded data by depth to nodal values by cap.'''

    logging.info('start track_grids_to_cap_list' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    cap_node = pid_d['cap_node']
    coor_cap_names = func_d['coor_cap_names']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    nodez = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']
    rm_list = func_d['rm_list']

    # grdtrack and store temperature in a cap data list
    value_by_cap = [[0]*cap_node for ii in range( nproc_surf )]
    track_file = 'track_grids_to_cap_list_track.xyz'
    if not track_file in rm_list: rm_list.append( track_file )
    for zz, grid in enumerate( grid_list ):
        for cc, cap_name in enumerate( coor_cap_names ):
            data = value_by_cap[cc]
            if grid is not None:
                cmd = '%(cap_name)s -G%(grid)s -fg' % vars()
                callgmt( 'grdtrack', cmd, '' , '>', track_file )
                value = np.loadtxt( track_file, usecols=(2,), unpack=True )
                value = np.clip( value, min, max )
                value = np.around( value, decimals=6 ).tolist()
            else: value = [ background for cc in range( nodex*nodey ) ]
            for nn, entry in enumerate( value ):
                data[ zz+nn*nodez ] = entry

    return value_by_cap

#=====================================================================
#=====================================================================
#=====================================================================
def citcom_data_export( master ):

    '''Main func_dtion to export CitcomS data.'''

    logging.info('start citcom_data_export' )

    control_d = master['control_d']
    pid_d = master['pid_d']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    OUTPUT_BVEL = control_d['OUTPUT_BVEL']
    OUTPUT_IVEL = control_d['OUTPUT_IVEL']
    OUTPUT_LITH_AGE = control_d['OUTPUT_LITH_AGE']
    OUTPUT_TEMP = control_d['OUTPUT_TEMP']
    OUTPUT_TEMP_IC = control_d['OUTPUT_TEMP_IC']
    OUTPUT_TRAC_IC = control_d['OUTPUT_TRAC_IC']
    SYNTHETIC = control_d['SYNTHETIC']
    
    # bvel (regional only) export
    if SYNTHETIC and OUTPUT_BVEL:
        output_regional_bvel( master )

    # ivel (also builds data since it is only used for export)
    if OUTPUT_IVEL:
        output_ivel( master )

    # lithosphere age export (regional and global)
    if OUTPUT_LITH_AGE:
        output_lith_age( master )

    # initial condition export (regional and global)
    if OUTPUT_TEMP_IC:
        output_initial_condition( master )

    # usual history (temperature) export
    if OUTPUT_TEMP:
        output_history( master )
    
    # tracer / composition generation
    if OUTPUT_TRAC_IC:
        output_tracer( master )

#=====================================================================
#=====================================================================
#=====================================================================
def get_slab_age_file( master, gplates_xy_filename ):

    '''Determine (thermal) age of slab segments from age grids.'''

    control_d = master['control_d']
    func_d = master['func_d']
    afile_1 = control_d['afile_1']
    res = control_d['gplates_line_resolution_km'] # in km
    rm_list = func_d['rm_list']

    if verbose: print( now(), 'get_slab_age_file:' )

    # find slab data from GPML data
    out_name = gplates_xy_filename.rstrip('xy') + 'head.xy'
    rm_list.append( out_name )
    in_name = Core_Util.get_slab_data( control_d, gplates_xy_filename, out_name )

    # get thermal ages from modified age grid (without mask)
    out_name = out_name.rstrip('xy') + 'age.xy'
    rm_list.append( out_name )
    out_name2 = Core_Util.find_value_on_line( in_name,
        control_d['afile_1'], out_name )

    return out_name2

#=====================================================================
#=====================================================================
#=====================================================================
def get_global_data( master_d ):

    '''Determine slab header data and thermal ages along the
       subduction zones.'''

    logging.info('start get_global_data' )

    # dictionaries
    control_d = master_d['control_d']
    func_d = master_d['func_d']

    # parameters
    FLAT_SLAB = control_d['FLAT_SLAB']
    FLAT_SLAB_RAMP = control_d['FLAT_SLAB_RAMP']
    grid_dir = control_d['grid_dir']
    OUTPUT_IVEL = control_d['OUTPUT_IVEL']
    sub_file = control_d['sub_file']
    rm_list = func_d['rm_list']

    # for subduction zone slabs
    slab_age_xyz = get_slab_age_file( master_d, control_d['sub_file'] )

    if OUTPUT_IVEL:
        # copy for ivels
        ivel_slab_age_xyz = slab_age_xyz.rstrip('xy') + 'ivel.xy'
        cmd = 'cp %(slab_age_xyz)s %(ivel_slab_age_xyz)s' % vars()
        subprocess.call( cmd, shell=True )
        func_d['ivel_slab_age_xyz'] = ivel_slab_age_xyz

    if FLAT_SLAB:
        # append leading edge slabs to subduction zone slabs
        lead_age_xyz = get_slab_age_file( master_d, control_d['flat_slab_leading_file'] )
        cmd = 'cat ' + lead_age_xyz + ' >> ' + slab_age_xyz
        subprocess.call( cmd, shell=True )
        if verbose: print( now(), cmd )
        flat_age_xyz = None

        if FLAT_SLAB_RAMP:
            # construct flat slab age depth file
            out_name = grid_dir + '/flat_slab_depth.grd'
            rm_list.append( out_name )
            Core_Util.make_flat_slab_depth_grd( master_d, out_name )
            flat_age_xyz = grid_dir + '/flat_slab_age_depth.xyz'
            rm_list.append( flat_age_xyz )
            Core_Util.make_flat_slab_age_depth_xyz( master_d, out_name,
                flat_age_xyz )

        # XXX DJB TODO - make a summary figure of flat slab?

    else:
        flat_age_xyz = None

    # pass values using function dictionary
    func_d['flat_age_xyz'] = flat_age_xyz
    func_d['slab_age_xyz'] = slab_age_xyz

#=====================================================================
#=====================================================================
#=====================================================================
def include_adiabat( master, master_grid, kk):

    '''Include a simple linear temperature increase across the whole
       mantle to model the effect of the mantle adiabat.  This is only
       applicable to compressible (TALA) or extended-Boussinesq
       models.'''

    if verbose: print( now(), 'include_adiabat:' )

    coor_d = master['coor_d']
    control_d = master['control_d']
    pid_d = master['pid_d']
    adiabat_temp_drop = control_d['adiabat_temp_drop']
    radius_inner = pid_d['radius_inner']
    radius_outer = pid_d['radius_outer']

    depth = coor_d['depth'][kk]

    # include linear adiabat
    add_temp = adiabat_temp_drop * depth/(radius_outer-radius_inner)
    cmd = master_grid + ' ' + str(add_temp) + ' ADD'
    callgmt( 'grdmath', cmd, '', '=', master_grid )
    
    return master_grid

#=====================================================================
#=====================================================================
#=====================================================================
def include_lithosphere( master, master_grid, kk ):

    '''Include the thermal profile of the lithosphere according to the
       age grid or a constant age.  N.B. This function actually
       computes the lithosphere temperature without considering the
       temperature drop across the lithosphere.  The temperature drop
       scaling is introduced by the multiplication with the
       master_grid in the final GMT call.'''

    if verbose: print( now(), 'include_lithosphere:' )

    control_d = master['control_d']
    coor_d = master['coor_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = str(control_d['age'])
    grid_dir = control_d['grid_dir']
    scalet = pid_d['scalet']
    rm_list = func_d['rm_list']
    suffix = control_d['suffix']
    utbl_age = control_d['utbl_age']
    UTBL_AGE_GRID = control_d['UTBL_AGE_GRID']

    # do nothing and exit
    depth_km = coor_d['depth_km'][kk]
    if depth_km > control_d['lith_depth_gen']: return master_grid, None

    # else include lithosphere
    depth = coor_d['depth'][kk]

    # use age grids
    if UTBL_AGE_GRID:
        if verbose: print( now(), 'Using age grids to construct lithosphere' )
        afile_2 = control_d['afile_2']
        grid = grid_dir + '/lithblend' + suffix + 'grd'
        rm_list.append( grid )
        cmd1 = 'cp %(afile_2)s %(grid)s' % vars()
        subprocess.call( cmd1, shell=True )
        cmd2 = '%(grid)s %(scalet)s DIV SQRT' % vars()
        callgmt( 'grdmath', cmd2, '', '=', grid )
        cmd3 = '%(depth)s 0.5 MUL %(grid)s DIV ERF' % vars()
        callgmt( 'grdmath', cmd3, '', '=', grid )

    # constant age
    else:
        print( now(), 'WARNING: constant age lithosphere' )
        print( now(), 'WARNING: utbl_age =', utbl_age, 'Ma' )
        val = Core_Util.erf((0.5*depth)/np.sqrt(utbl_age/scalet))
        grid = str( val )

    cmd = master_grid + ' ' + grid + ' MUL'
    callgmt( 'grdmath', cmd, '', '=', master_grid )
    
    return master_grid, grid

#=====================================================================
#=====================================================================
#=====================================================================
def include_blob( master, master_grids, master_node_points, kk ):

    '''Include thermal blobs.'''

    if verbose: print( now(), 'include_blob:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    rm_list = func_d['rm_list']
    coor_d = master['coor_d']
    r = 1.0 - coor_d['depth'][kk]
    R = pid_d['radius']
    curr_age = control_d['age']
    
    blob_center_lon = control_d['blob_center_lon']
    blob_center_lat = control_d['blob_center_lat']
    blob_center_depth= control_d['blob_center_depth']
    blob_radius = control_d['blob_radius']
    blob_birth_age = control_d['blob_birth_age']
    blob_dT = control_d['blob_dT']
    blob_profile = control_d['blob_profile']

    if((type(blob_center_lon) is not list) and \
       (type(blob_center_lat) is not list) and \
       (type(blob_center_depth) is not list) and \
       (type(blob_radius) is not list) and \
       (type(blob_birth_age) is not list) and \
       (type(blob_dT) is not list) and \
       (type(blob_profile) is not list)):
        
        blob_center_lon = [float(blob_center_lon)]
        blob_center_lat = [float(blob_center_lat)]
        blob_center_depth = [float(blob_center_depth)]
        blob_radius = [float(blob_radius)]
        blob_birth_age = [float(blob_birth_age)]
        blob_dT = [float(blob_dT)]
        blob_profile = [str(blob_profile)]
    else:
        if((len(blob_center_lon) != len(blob_center_lat)) or \
           (len(blob_center_lon) != len(blob_center_depth)) or \
           (len(blob_center_lon) != len(blob_radius)) or \
           (len(blob_center_lon) != len(blob_birth_age)) or \
           (len(blob_center_lon) != len(blob_dT)) or \
           (len(blob_center_lon) != len(blob_profile))):
            print( now(), 'ERROR: inconsistent parameterization for Blobs' )
            sys.exit(1)
        #end if
        blob_center_lon = [float(i) for i in blob_center_lon]
        blob_center_lat = [float(i) for i in blob_center_lat]
        blob_center_depth = [float(i) for i in blob_center_depth]
        blob_radius = [float(i) for i in blob_radius]
        blob_birth_age = [float(i) for i in blob_birth_age]
        blob_dT = [float(i) for i in blob_dT]        
        blob_profile = [str(i) for i in blob_profile]        
    #end if

    # check if birth ages fall within age-range and issue error message otherwise.
    age_start = max( control_d['age_start'], control_d['age_end'] )
    age_end = min( control_d['age_end'], control_d['age_start'] )
    age_loop = list( range( age_end, age_start+1 ) )
    age_loop_set = set(age_loop)
    blob_birth_age_set = set(blob_birth_age)

    if(not (blob_birth_age_set.issubset(age_loop_set))):
        print( now(), '''ERROR: inconsistent parameterization for Blobs. Blob birth 
                       ages are outside the age-range [age_start, age_end].''' )
        sys.exit(1)
    #end if

    out_name = master_grids[0].rstrip('grd') + 'xyz'
    if(master_node_points is None):
        rm_list.append( out_name )
        cmd = master_grids[0] + ' -S'
        
        Core_GMT.callgmt( 'grd2xyz', cmd, '', '>', out_name )
        master_node_points = []
        lon, lat, val = np.loadtxt( out_name, unpack=True )
        for j in range(0, len(lon)):
            currNodeRTP = [1, (90-lat[j])/180*math.pi, lon[j]/180*math.pi]
            currNodeXYZ = Core_Util.spher2cart_coord(currNodeRTP[0], \
                                                     currNodeRTP[1], \
                                                     currNodeRTP[2])
            master_node_points.append([lon[j], \
                                       lat[j], \
                                       currNodeRTP[2], \
                                       currNodeRTP[1], \
                                       currNodeXYZ[0], \
                                       currNodeXYZ[1], \
                                       currNodeXYZ[2]])
            #end if
        #end for        
    #end if
    
    # blob_profile_functions
    def blob_constant_profile(dist, radius, amp): return amp
    def blob_exponential_profile(dist, radius, amp):
        return amp * math.exp(-1.0*dist/radius)
    #end function
    def blob_gaussian1_profile(dist, radius, amp):
        return amp * math.exp(-1.0*math.pow(dist/radius,2.))
    #end function
    def blob_gaussian2_profile(dist, radius, amp):
        return amp * (1-math.pow(dist/radius,2.)) * \
                    math.exp(-1.0*math.pow(dist/radius,2.))
    #end function

    profileFunctionPointers = {"constant":blob_constant_profile, \
                               "exponential":blob_exponential_profile, \
                               "gaussian1":blob_gaussian1_profile, \
                               "gaussian2":blob_gaussian2_profile}

    val = [0]*len(master_node_points)
    sten = [0]*len(master_node_points)

    grid_vals = [val, sten]
    for i in range(0, len(blob_center_lon)):

        # skip blob if birth age != curr_age
        if(blob_birth_age[i] != curr_age): 
            #print ('skipping blob at %f' % (blob_birth_age[i]))
            continue
        #end if

        blobCenterRTP = [(R-blob_center_depth[i]*1e3)/R, \
                         blob_center_lat[i]/180*math.pi, \
                         blob_center_lon[i]/180*math.pi] 
        blobCenterXYZ = Core_Util.spher2cart_coord(blobCenterRTP[0], \
                                                   blobCenterRTP[1], \
                                                   blobCenterRTP[2])
        blobRadiusND = blob_radius[i]/R*1e3
        blobRadiusND2 = blobRadiusND * blobRadiusND
        
        blobProfileFunction = profileFunctionPointers[blob_profile[i]]

        angleSubtendedByBase = blobRadiusND / blobCenterRTP[0]
        for j in range(0, len(master_node_points)):
            
            if(math.fabs(blobCenterRTP[1] - master_node_points[j][3]) \
                > angleSubtendedByBase):
                if(math.fabs(blobCenterRTP[2] - master_node_points[j][2]) \
                    > angleSubtendedByBase): 
                    continue
                #end if
            #end if

            test_point = [master_node_points[j][4] * r, \
                          master_node_points[j][5] * r, \
                          master_node_points[j][6] * r]

            dx = blobCenterXYZ[0] - test_point[0]
            dy = blobCenterXYZ[1] - test_point[1]
            dz = blobCenterXYZ[2] - test_point[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if(dist<=blobRadiusND):
                val[j] += blobProfileFunction(dist, blobRadiusND, \
                                              blob_dT[i])
                sten[j] = 1.;
            #end if
        #end for
    #end for

    for igrid in range(len(master_grids)):
        out_file = open(out_name, 'w')
        for i in range(0, len(master_node_points)):
            line = ('%lf %lf %lf\n')%(master_node_points[i][0], \
                                      master_node_points[i][1], grid_vals[igrid][i])
            out_file.write(line)
        #end for
        out_file.close()
        
        interim_grid = master_grids[igrid].rstrip('grd') + 'interim.grd'
        rm_list.append( interim_grid )
        cmd = out_name + ' -R'+master_grids[igrid] + ' -G'+interim_grid
        
        Core_GMT.callgmt( 'xyz2grd', cmd)
        cmd = master_grids[igrid] + ' ' + interim_grid + ' ADD'
        callgmt( 'grdmath', cmd, '', '=', master_grids[igrid] )
    #end for
    
    return master_grids[0], master_grids[1], master_node_points
    
#=====================================================================
#=====================================================================
#=====================================================================
def include_silo( master, master_grids, master_node_points, kk ):

    '''Include thermal silos.'''

    if verbose: print( now(), 'include_silo:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    rm_list = func_d['rm_list']
    coor_d = master['coor_d']
    r = 1.0 - coor_d['depth'][kk]
    R = pid_d['radius']
    curr_age = control_d['age']
    
    silo_base_center_lon = control_d['silo_base_center_lon']
    silo_base_center_lat = control_d['silo_base_center_lat']
    silo_base_center_depth= control_d['silo_base_center_depth']
    silo_radius = control_d['silo_radius']
    silo_cylinder_height = control_d['silo_cylinder_height']
    silo_birth_age = control_d['silo_birth_age']
    silo_dT = control_d['silo_dT']
    silo_profile = control_d['silo_profile']

    if((type(silo_base_center_lon) is not list) and \
       (type(silo_base_center_lat) is not list) and \
       (type(silo_base_center_depth) is not list) and \
       (type(silo_radius) is not list) and \
       (type(silo_cylinder_height) is not list) and \
       (type(silo_birth_age) is not list) and \
       (type(silo_dT) is not list) and \
       (type(silo_profile) is not list)):
        
        silo_base_center_lon = [float(silo_base_center_lon)]
        silo_base_center_lat = [float(silo_base_center_lat)]
        silo_base_center_depth = [float(silo_base_center_depth)]
        silo_radius = [float(silo_radius)]
        silo_cylinder_height = [float(silo_cylinder_height)]
        silo_birth_age = [float(silo_birth_age)]
        silo_dT = [float(silo_dT)]
        silo_profile = [str(silo_profile)]
    else:
        if((len(silo_base_center_lon) != len(silo_base_center_lat)) or \
           (len(silo_base_center_lon) != len(silo_base_center_depth)) or \
           (len(silo_base_center_lon) != len(silo_radius)) or \
           (len(silo_base_center_lon) != len(silo_cylinder_height)) or \
           (len(silo_base_center_lon) != len(silo_birth_age)) or \
           (len(silo_base_center_lon) != len(silo_dT)) or \
           (len(silo_base_center_lon) != len(silo_profile))):
            print( now(), 'ERROR: inconsistent parameterization for Silos' )
            sys.exit(1)
        #end if
        silo_base_center_lon = [float(i) for i in silo_base_center_lon]
        silo_base_center_lat = [float(i) for i in silo_base_center_lat]
        silo_base_center_depth = [float(i) for i in silo_base_center_depth]
        silo_radius = [float(i) for i in silo_radius]
        silo_cylinder_height = [float(i) for i in silo_cylinder_height]
        silo_birth_age = [float(i) for i in silo_birth_age]
        silo_dT = [float(i) for i in silo_dT]        
        silo_profile = [str(i) for i in silo_profile]        
    #end if
    
    # check if birth ages fall within age-range and issue error message otherwise.
    age_start = max( control_d['age_start'], control_d['age_end'] )
    age_end = min( control_d['age_end'], control_d['age_start'] )
    age_loop = list( range( age_end, age_start+1 ) )
    age_loop_set = set(age_loop)
    silo_birth_age_filtered = []
    for i in silo_birth_age:
        if(i != -1): silo_birth_age_filtered.append(i)
    #end for
    silo_birth_age_set = set(silo_birth_age_filtered)

    if((len(silo_birth_age_set)>0) and (not (silo_birth_age_set.issubset(age_loop_set)))):
        print( now(), '''ERROR: inconsistent parameterization for Silos. Silo birth 
                       ages are outside the age-range [age_start, age_end].''' )
        sys.exit(1)
    #end if
    
    out_name = master_grids[0].rstrip('grd') + 'xyz'
    if(master_node_points is None):
        rm_list.append( out_name )
        cmd = master_grids[0] + ' -S'
        
        Core_GMT.callgmt( 'grd2xyz', cmd, '', '>', out_name )
        master_node_points = []
        lon, lat, val = np.loadtxt( out_name, unpack=True )
        for j in range(0, len(lon)):
            currNodeRTP = [1, (90-lat[j])/180*math.pi, lon[j]/180*math.pi]
            currNodeXYZ = Core_Util.spher2cart_coord(currNodeRTP[0], \
                                                     currNodeRTP[1], \
                                                     currNodeRTP[2])
            master_node_points.append([lon[j], \
                                       lat[j], \
                                       currNodeRTP[2], \
                                       currNodeRTP[1], \
                                       currNodeXYZ[0], \
                                       currNodeXYZ[1], \
                                       currNodeXYZ[2]])
            #end if
        #end for        
    #end if
    
    # silo_profile_functions
    def silo_constant_profile(dist, radius, amp): return amp
    def silo_exponential_profile(dist, radius, amp):
        return amp * math.exp(-1.0*dist/radius)
    #end function
    def silo_gaussian1_profile(dist, radius, amp):
        return amp * math.exp(-1.0*math.pow(dist/radius,2.))
    #end function
    def silo_gaussian2_profile(dist, radius, amp):
        return amp * (1-math.pow(dist/radius,2.)) * \
                    math.exp(-1.0*math.pow(dist/radius,2.))
    #end function

    profileFunctionPointers = {"constant":silo_constant_profile, \
                               "exponential":silo_exponential_profile, \
                               "gaussian1":silo_gaussian1_profile, \
                               "gaussian2":silo_gaussian2_profile}


    val = [0]*len(master_node_points)
    sten = [0]*len(master_node_points)

    grid_vals = [val, sten]
    for i in range(0, len(silo_base_center_lon)):

        # skip silo if birth age != curr_age, unless birth_age is set to -1
        if((silo_birth_age[i] != curr_age) and (silo_birth_age[i] != -1)):
            #print ('skipping silo at %f' % (silo_birth_age[i]))
            continue
        #end if

        siloBaseCenterRTP = [(R-silo_base_center_depth[i]*1e3)/R, \
                         silo_base_center_lat[i]/180*math.pi, \
                         silo_base_center_lon[i]/180*math.pi] 
        siloBaseCenterXYZ = Core_Util.spher2cart_coord(siloBaseCenterRTP[0], \
                                                   siloBaseCenterRTP[1], \
                                                   siloBaseCenterRTP[2])
        
        siloAxisUnitVectorCart = Core_Util.spher2cart_vector(1., 0, 0,  \
                                                             siloBaseCenterRTP[0], \
                                                             siloBaseCenterRTP[1], \
                                                             siloBaseCenterRTP[2])
        siloTopCenterXYZ = [0]*3
        for dim in range(0, 3): siloTopCenterXYZ[dim] = siloBaseCenterXYZ[dim] + \
                                                        silo_cylinder_height[i] * 1e3 *\
                                                        siloAxisUnitVectorCart[dim] / R
        siloRadiusND = silo_radius[i]/R*1e3
        siloRadiusND2 = siloRadiusND * siloRadiusND
        siloHeightND2 = silo_cylinder_height[i]/R*1e3*silo_cylinder_height[i]/R*1e3

        siloProfileFunction = profileFunctionPointers[silo_profile[i]]

        angleSubtendedByBase = siloRadiusND / siloBaseCenterRTP[0]
        for j in range(0, len(master_node_points)):
            
            if(math.fabs(siloBaseCenterRTP[1] - master_node_points[j][3]) \
                > angleSubtendedByBase):
                if(math.fabs(siloBaseCenterRTP[2] - master_node_points[j][2]) \
                    > angleSubtendedByBase): 
                    continue
                #end if
            #end if

            test_point = [master_node_points[j][4] * r, \
                          master_node_points[j][5] * r, \
                          master_node_points[j][6] * r]
            dist = Core_Util.is_coord_within_silo(siloBaseCenterXYZ, \
                                              siloTopCenterXYZ, \
                                              siloHeightND2, \
                                              siloRadiusND2, \
                                              test_point)
            if(dist > -1):
                val[j] += silo_dT[i]
                sten[j] = siloProfileFunction(dist, siloRadiusND, 1.0)
            #end if
        #end for
    #end for

    for igrid in range(len(master_grids)):
        out_file = open(out_name, 'w')
        for i in range(0, len(master_node_points)):
            line = ('%lf %lf %lf\n')%(master_node_points[i][0], \
                                      master_node_points[i][1], grid_vals[igrid][i])
            out_file.write(line)
        #end for
        out_file.close()
        
        interim_grid = master_grids[igrid].rstrip('grd') + 'interim.grd'
        rm_list.append( interim_grid )
        cmd = out_name + ' -R'+master_grids[igrid] + ' -G'+interim_grid
        
        Core_GMT.callgmt( 'xyz2grd', cmd)
        cmd = master_grids[igrid] + ' ' + interim_grid + ' ADD'
        callgmt( 'grdmath', cmd, '', '=', master_grids[igrid] )
    #end for
    
    return master_grids[0], master_grids[1], master_node_points

#=====================================================================
#=====================================================================
#=====================================================================
def include_ltbl( master, master_grid, kk ):

    '''Include a lower thermal boundary layer.  CMB temperature = 1 
       (by definition).'''

    if verbose: print( now(), 'include_ltbl:' )

    coor_d = master['coor_d']
    control_d = master['control_d']
    pid_d = master['pid_d']
    BUILD_ADIABAT = control_d['BUILD_ADIABAT']
    ltbl_age = control_d['ltbl_age']
    radius_inner = pid_d['radius_inner']
    radius_outer = pid_d['radius_outer']
    radius = pid_d['radius']
    scalet = pid_d['scalet']
    temperature_cmb = control_d['temperature_cmb']
    temperature_mantle = control_d['temperature_mantle']

    if BUILD_ADIABAT:
        adiabat_temp_drop = control_d['adiabat_temp_drop']
        temperature_mantle += adiabat_temp_drop

    # do nothing and exit
    depth = coor_d['depth'][kk]
    height_above_cmb = radius_outer - radius_inner - depth
    if height_above_cmb > 0.1: return master_grid

    # else include ltbl
    arg = 0.5 * height_above_cmb * 1 / np.sqrt(ltbl_age/scalet)
    add_temp = (temperature_cmb-temperature_mantle) * Core_Util.erfc( arg )

    cmd = master_grid + ' ' + str(add_temp) + ' ADD'
    callgmt( 'grdmath', cmd, '', '=', master_grid )

    return master_grid

#=====================================================================
#=====================================================================
#=====================================================================
def make_tracer_summary_postscript( master_d ):

    '''Create a summary postscript of tracer distribution.'''

    logging.info('make_tracer_summary_postscript' )

    # dictionaries
    control_d = master_d['control_d']
    func_d = master_d['func_d']

    # parameters
    afile_4 = control_d['afile_4']
    age = control_d['age']
    grid_dir = control_d['grid_dir']
    ps_dir = control_d['ps_dir']
    rm_list = func_d['rm_list']
    stencil_values = control_d['stencil_values']
    sub_file_left = control_d['sub_file_left']
    sub_file_right = control_d['sub_file_right']

    make_dir( ps_dir )

    ps = ps_dir + '/tracer.%(age)sMa.ps' % vars()

    # start postscript
    callgmt( 'gmtset', 'PAGE_ORIENTATION', '', '', 'portrait' )
    callgmt( 'gmtset', 'LABEL_FONT_SIZE', '', '', '12' )
    callgmt( 'gmtset', 'LABEL_FONT', '', '', '4' )
    callgmt( 'gmtset', 'LABEL_OFFSET', '', '', '0.02' )
    callgmt( 'gmtset', 'ANNOT_FONT_SIZE_PRIMARY', '', '', '10p' )
    callgmt( 'gmtset', 'ANNOT_FONT_PRIMARY', '', '', '4' )
    gmt_base_options = Core_GMT.start_postscript( ps ) 
    additional_options = {key: control_d[key] for key in control_d if
        key=='J' or key=='R'}
    gmt_base_options.update( additional_options )
    B = control_d['B']

    # title
    stdin = '8.0 10.5 14 0 4 MR Age = %(age)s Ma\n' % vars()
    stdin += '8.0 10.25 14 0 4 MR Tracer stencil value\nEOF' % vars()
    cmd = '-R0/8.5/0/11 -Jx1.0 -K -O'
    callgmt('pstext', cmd, '','<< EOF >>', ps + '\n' + stdin)

    # cpt
    stencil_cpt = grid_dir + '/tracer_stencil.cpt'
    rm_list.append( stencil_cpt )
    num_stencil_values = len(list(control_d['stencil_values']))
    zmin = -num_stencil_values-0.5
    zmax = 0.5
    cmd = '-Cjet -T%(zmin)s/%(zmax)s/1' % vars()
    callgmt( 'makecpt', cmd, '', '>', stencil_cpt )

    cmd = '%(afile_4)s -C%(stencil_cpt)s -B%(B)s -Xa0.5 -Ya8' % vars()
    callgmt( 'grdimage', cmd, gmt_base_options, '>>', ps )

    # overlay subduction zones
    G = 'black'
    W = '3/black'
    if sub_file_left:
        cmd = sub_file_left + ' -Sf0.2i/0.05ilt -G%(G)s -W%(W)s -m \
-Xa0.5 -Ya8' % vars()
        callgmt( 'psxy', cmd, gmt_base_options, '>>', ps )
    if sub_file_right:
        cmd = sub_file_right + ' -Sf0.2i/0.05irt -G%(G)s -W%(W)s -m \
-Xa0.5 -Ya8' % vars()
        callgmt( 'psxy', cmd, gmt_base_options, '>>', ps )

    cmd = '-C%(stencil_cpt)s -D1.5/7.6/1.5/0.125h -K -O \
        -B1:"Stencil value":' % vars() # psscale
    callgmt( 'psscale', cmd, '', '>>', ps )

    # end postscript
    Core_GMT.end_postscript( ps )

#=====================================================================
#=====================================================================
#=====================================================================
def make_summary_postscript( master, master_grids, kk,
                      temperature_mantle_grid, slab_grid, lith_grid ):

    '''Create a summary postscript for this depth.'''

    logging.info('start make_summary_postscript') 

    coor_d = master['coor_d']
    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    # afile_2 might not exist if UTBL_AGE_GRID is False
    age_grid = control_d.get('afile_2', None)
    grid_dir = control_d['grid_dir']
    ps_dir = control_d['ps_dir']
    rm_list = func_d['rm_list']
    stencil_min = control_d['stencil_min']
    stencil_max = control_d['stencil_max']
    SYNTHETIC = control_d['SYNTHETIC']
    temperature_min = control_d['temperature_min']
    # always temperature_max = 1.0 for plotting so plots with
    # and without a LTBL can easily be compared
    sub_file_left = control_d.get('sub_file_left', None)
    sub_file_right = control_d.get('sub_file_right', None)
    temperature_max = 1.0 #control_d['temperature_max']
    utbl_age_max = control_d['utbl_age_max']

    make_dir( ps_dir )

    temp_grid = master_grids[0]
    sten_grid = master_grids[1]

    ps = ps_dir + '/' + temp_grid.split('/')[-1].rstrip('grd') + 'ps'
    age = control_d['age']
    depth_km = int(coor_d['depth_km'][kk])

    # start postscript
    callgmt( 'gmtset', 'PAGE_ORIENTATION', '', '', 'portrait' )
    callgmt( 'gmtset', 'LABEL_FONT_SIZE', '', '', '12' )
    callgmt( 'gmtset', 'LABEL_FONT', '', '', '4' )
    callgmt( 'gmtset', 'LABEL_OFFSET', '', '', '0.02' )
    callgmt( 'gmtset', 'ANNOT_FONT_SIZE_PRIMARY', '', '', '10p' )
    callgmt( 'gmtset', 'ANNOT_FONT_PRIMARY', '', '', '4' )
    gmt_base_options = Core_GMT.start_postscript( ps )
    additional_options = {key: control_d[key] for key in control_d if
        key=='J' or key=='R'}
    gmt_base_options.update( additional_options )
    B = control_d['B']

    # title
    stdin = '8.0 10.5 14 0 4 MR Age = %(age)s Ma\n' % vars()
    stdin += '8.0 10.25 14 0 4 MR Depth = %(depth_km)s km\nEOF' % vars()
    cmd = '-R0/8.5/0/11 -Jx1.0 -K -O'
    callgmt('pstext', cmd, '','<< EOF >>', ps + '\n' + stdin)

    # thermal age of lithosphere
    if age_grid:
        age_cpt = grid_dir + '/age.cpt'
        if not age_cpt in rm_list: rm_list.append( age_cpt )
        if SYNTHETIC:
            cmd = '-Crainbow -D -T0/120/5 -I'
        else:
            cpt_max = int( np.ceil( utbl_age_max/10 ) * 10 )
            cmd = '-Crainbow -T0/%(cpt_max)s/10 -I' % vars()
        callgmt( 'makecpt', cmd, '', '>', age_cpt )
        cmd = '%(age_grid)s -C%(age_cpt)s -B%(B)s -Xa0.5 -Ya8' % vars()
        callgmt( 'grdimage', cmd, gmt_base_options, '>>', ps )

    # overlay velocity arrows
    if SYNTHETIC:
        # overlay velocity
        filename = func_d['velocity_map']
        G = 'darkgrey'
        S = 'V0.015i/0.06i/0.05i'
        X = 'a0.5'
        Y = 'a8' 
        cmd = '%(filename)s -G%(G)s -S%(S)s -X%(X)s -Y%(Y)s' % vars()
        callgmt( 'psxy', cmd, gmt_base_options, '>>', ps ) 

        # velocity scale bar
        velocity_scale = control_d['velocity_scale'] / 4
        stdin = '''# velocity vector
S 0.125 v 0.25/0.015/0.06/0.05 0/0/0 1,black 0.27i %(velocity_scale).0f cm/yr\nEOF''' % vars()
        C = '0.05i'
        D = '3.5/7.6/0.875/0.20/TC'
        J = 'x1.0'
        R = '0/8.5/0/11'
        cmd = '-J%(J)s -R%(R)s -C%(C)s -D%(D)s -K -O' % vars()
        callgmt( 'pslegend', cmd, '', '<< EOF >>', ps + '\n' +stdin )

    # overlay subduction zones
    G = 'black'
    W = '3/black'
    if sub_file_left:
        cmd = sub_file_left + ' -Sf0.2i/0.05ilt -G%(G)s -W%(W)s -m \
-Xa0.5 -Ya8' % vars()
        callgmt( 'psxy', cmd, gmt_base_options, '>>', ps )
    if sub_file_right:
        cmd = sub_file_right + ' -Sf0.2i/0.05irt -G%(G)s -W%(W)s -m \
-Xa0.5 -Ya8' % vars()
        callgmt( 'psxy', cmd, gmt_base_options, '>>', ps ) 

    if age_grid:
        if SYNTHETIC:
            cmd = '-C%(age_cpt)s -D1.5/7.6/1.5/0.125h -K -O \
-B30:"Seafloor age (Ma)":' % vars() # psscale
        else:
            cmd = '-C%(age_cpt)s -D1.5/7.6/1.5/0.125h -K -O \
-B50:"Thermal age":/:"Ma":' % vars() # psscale
        callgmt( 'psscale', cmd, '', '>>', ps )

    stdin = '0.25 10.5 12 0 4 ML Lithosphere age (Ma)\nEOF'
    cmd = '-R0/8.5/0/11 -Jx1.0 -K -O'
    callgmt('pstext', cmd, '','<< EOF >>', ps + '\n' + stdin)


    temp_cpt = grid_dir + '/temp.cpt'
    if not temp_cpt in rm_list: rm_list.append( temp_cpt )
    cmd = '-Cjet -D -T%f/%f/0.025' % (temperature_min, temperature_max)
    callgmt( 'makecpt', cmd, '', '>', temp_cpt )
    cmd = '-C%(temp_cpt)s -D6.75/5.1/1.5/0.125h -K -O \
 -B0.2:"Temperature":/:"Non-dim":' % vars() # psscale
    callgmt( 'psscale', cmd, '', '>>', ps )

    # lithosphere
    if lith_grid:
        scale_lith_grid = grid_dir + '/lithscale_for_plot_only.grd'
        if not scale_lith_grid in rm_list:
            rm_list.append( scale_lith_grid )
        cmd = lith_grid + ' ' + temperature_mantle_grid + ' MUL'
        callgmt( 'grdmath', cmd, '', '=',  scale_lith_grid )
        cmd = scale_lith_grid + ' -C%(temp_cpt)s -B%(B)s -Xa3.0 -Ya5.5' % vars()
        callgmt( 'grdimage', cmd, gmt_base_options, '>>', ps )
        stdin = '8.25 8 12 0 4 MR Lithosphere temp\nEOF'
        cmd = '-R0/8.5/0/11 -Jx1.0 -K -O'
        callgmt('pstext', cmd, '','<< EOF >>', ps + '\n' + stdin)

    # thermal slabs
    if slab_grid:
        cmd = slab_grid + ' -C%(temp_cpt)s -B%(B)s -Xa0.5 -Ya3.0' % vars()
        callgmt( 'grdimage', cmd, gmt_base_options, '>>', ps )
        W = '2/black'
        if sub_file_left:
            cmd = sub_file_left + ' -W%(W)s -m \
-Xa0.5 -Ya3' % vars()
            callgmt( 'psxy', cmd, gmt_base_options, '>>', ps )
        if sub_file_right:
            cmd = sub_file_right + ' -W%(W)s -m \
-Xa0.5 -Ya3' % vars()
            callgmt( 'psxy', cmd, gmt_base_options, '>>', ps )
        stdin = '0.25 5.5 12 0 4 ML Slab temp\nEOF'
        cmd = '-R0/8.5/0/11 -Jx1.0 -K -O'
        callgmt('pstext', cmd, '','<< EOF >>', ps + '\n' + stdin)

    # temperature (total)
    cmd = temp_grid + ' -C%(temp_cpt)s -B%(B)s -Xa3.0 -Ya0.5' % vars()
    callgmt( 'grdimage', cmd, gmt_base_options, '>>', ps )

    # stencil contour
    sten_cpt = grid_dir + '/sten.cpt'
    if not sten_cpt in rm_list: rm_list.append( sten_cpt )
    cmd = '-Cjet -T%f/%f/0.2' % (stencil_min, stencil_max)
    # to plot 0.5 contour only uncomment line below
    cmd = '-Cjet -T-0.1/1/0.6'
    callgmt( 'makecpt', cmd, '', '>', sten_cpt )
    cmd = sten_grid + ' -C%(sten_cpt)s -Xa3.0 -Ya0.5' % vars()
    callgmt( 'grdcontour', cmd, gmt_base_options, '>>', ps )

    stdin = '8.25 3 12 0 4 MR Total temp\nEOF'
    cmd = '-R0/8.5/0/11 -Jx1.0 -K -O'
    callgmt('pstext', cmd, '','<< EOF >>', ps + '\n' + stdin)

    # end postscript
    Core_GMT.end_postscript( ps )

    return ps

#=====================================================================
#=====================================================================
#=====================================================================
def output_history( master ):

    '''Export thermal slab history (``hist'') files, one per cap.'''

    if verbose: print( now(), 'output_history:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']

    age = control_d['age']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    hist_dir = control_d['hist_dir']
    IC = control_d['IC']
    model_name = control_d['model_name']
    sten_by_cap = func_d['sten_by_cap']
    temp_by_cap = func_d['temp_by_cap']

    outname = hist_dir + '/' + model_name + '.hist.dat' + str(age)
    outname2 = hist_dir + '/' + model_name + '.hist.dat' + str(age+1)

    if FULL_SPHERE:
        outname += '.#' # cap number suffix
        outname2 += '.#'

    # data to output in columns
    out_data = ( temp_by_cap, sten_by_cap )

    make_dir( hist_dir )
    Core_Citcom.write_cap_or_proc_list_to_files( pid_d, outname, 
                                              out_data, 'cap', False )

    if IC:
        Core_Citcom.write_cap_or_proc_list_to_files( pid_d, outname2,
                                              out_data, 'cap', False )

#=====================================================================
#=====================================================================
#=====================================================================
def output_tracer( master_d ):

    '''Output tracer initial condition.'''

    logging.info('start output_tracer' )

    # XXX DJB - for testing efficiency
    t0 = time.time() # start time

    # dictionaries
    control_d = master_d['control_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']
    pid_d = master_d['pid_d']

    # parameters
    age = control_d['age']
    CONTINENTAL_TYPES = control_d['CONTINENTAL_TYPES']
    DEEP_LAYER_TRACERS = control_d['DEEP_LAYER_TRACERS']
    model_name = control_d['model_name']
    NO_TRACER_REGION = control_d['NO_TRACER_REGION']
    if  pid_d['Solver'] == 'multigrid':
        nodex = pid_d['mgunitx'] * int(math.pow(2,pid_d['levels']-1)) * pid_d['nprocx'] + 1
        nodey = pid_d['mgunity'] * int(math.pow(2,pid_d['levels']-1)) * pid_d['nprocy'] + 1
        nodez = pid_d['mgunitz'] * int(math.pow(2,pid_d['levels']-1)) * pid_d['nprocz'] + 1
        if nodex != pid_d['nodex'] or nodey != pid_d['nodey'] or nodez != pid_d['nodez']:
            logging.warning('The equation nodex = 1 + nprocx + mgunitx + 2**(levels-1) must be satisfied!')
            logging.warning('The nodex, nodey and nodez have been modified to satisfy the equation.')
    else:
        nodex = pid_d['nodex']
        nodey = pid_d['nodey']
        nodez = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']
    PLOT_SUMMARY_POSTSCRIPT = control_d['PLOT_SUMMARY_POSTSCRIPT']
    radius_km = pid_d['radius_km']
    radius_outer = pid_d['radius_outer']
    rm_list = func_d['rm_list']
    trac_dir = control_d['trac_dir']
    tracer_no_ass_depth = control_d['tracer_no_ass_depth']
    tracers_per_element = pid_d['tracers_per_element']

    num = nodex*nodey*nodez*nproc_surf*tracers_per_element
    txt = 'tracers_per_element= %(tracers_per_element)s' % vars()
    if verbose: print( now(), txt )
    txt = 'total number of tracers= %(num)s' % vars()
    if verbose: print( now(), txt )

    r, theta, phi = Core_Util.generate_random_tracers( num )

    # remove tracers
    if NO_TRACER_REGION:
        no_tracer_min_depth = control_d['no_tracer_min_depth']
        no_tracer_max_depth = control_d['no_tracer_max_depth']
        txt = 'WARNING: removing tracers between '
        txt += '%(no_tracer_min_depth)s km and %(no_tracer_max_depth)s km depth' % vars()
        logging.warning( txt )
        max_rad = radius_outer - ( no_tracer_min_depth / radius_km )
        min_rad = radius_outer - ( no_tracer_max_depth / radius_km )
        index = np.where(((r > max_rad) | (r < min_rad)))
        r = r[index]
        theta = theta[index]
        phi = phi[index]
        # redefine number of tracers to truncated range
        num = np.size( index )
        logging.warning(f'total number of tracers reduced to {num}')

    # ambient tracers are flavor zero
    # N.B. this is reset if TRACER_NO_ASSIM is True
    flavor = np.zeros( num )

    # build continental tracers
    if CONTINENTAL_TYPES:
        # write out coordinates for grdtrack
        make_dir( trac_dir )
        filename = '%(trac_dir)s/tracer_lon_lat_for_grdtrack.xy' % vars()
        rm_list.append( filename )
        lon = np.degrees( phi )
        lat = 90 - np.degrees( theta )
        out_data = np.column_stack( (lon, lat) )
        txt = 'writing file= %(filename)s' % vars()
        if verbose: print( now(), txt )
        np.savetxt( filename, out_data, fmt='%.2f' )

        # set no assimilation regions (if present)
        # to NaN
        make_stencil_grid_to_build_tracers( master_d )
        afile_4 = control_d['afile_4']

        # get stencil values
        cmd = filename + ' -G' + afile_4
        track_file = '%(trac_dir)s/output_tracer_track.xyz' % vars()
        rm_list.append( track_file )
        callgmt( 'grdtrack', cmd,  '', '>', track_file )
        # grdtrack command produces negative non-integer values at
        # the edge of continental types due to interpolation
        sv = np.loadtxt( track_file, usecols=(2,), unpack=True )
        sv = np.around( sv, decimals=0 )
        # XXX DJB
        # here we could also np.clip the data (minimum value of
        # -num_stencil_values).  However, stencil values outside the range
        # of stencil_values are already initialized to zeroes (ambient flavor)
        # which is probably the safest fall-back anyway

        # check for no assimilation
        # indices of tracers within no assimilation regions and less than
        # tracer_no_ass_depth
        tracer_no_ass_radius = 1 - tracer_no_ass_depth/pid_d['radius']
        no_ass_indices = np.where((np.isnan(sv)) & (r > tracer_no_ass_radius))

        if np.size( no_ass_indices ):
            # for testing
            #lon_test = lon[no_ass_indices]
            #lat_test = lat[no_ass_indices]
            #format_test = '%.6f %.6f'
            #out_data_test = np.column_stack( (lon_test, lat_test) )
            #np.savetxt( 'tracer_no_assim_test.xy', out_data_test, fmt=format_test, comments='' )

            # must be AFTER any np.isnan calls since astype(int) does not
            # conserve NaN!
            sv = sv.astype(int)

            # redefine number of tracers and arrays
            sv = np.delete( sv, no_ass_indices)
            theta = np.delete( theta, no_ass_indices )
            phi = np.delete( phi, no_ass_indices )
            r = np.delete( r, no_ass_indices )
            num = np.size( sv )
            flavor = np.zeros( num )
            txt = 'WARNING: removing tracers within no assimilation regions'
            print( now(), txt )
            txt = 'WARNING: excluding tracers from surface to %(tracer_no_ass_depth)s km depth' % vars()
            print( now(), txt )
            txt = 'WARNING: total number of tracers reduced to %(num)s' % vars()
            print( now(), txt )


        # map stencil values and depths to tracer flavors
        # must keep list() so that single values are converted
        num_stencil_values = len(list(control_d['stencil_values']))

        for stencil_value in range( -num_stencil_values, 1 ):
            if verbose: print( now(), 'processing stencil_value= ', stencil_value )
            sten_val_abs = abs( stencil_value )
            suffix = '_stencil_value_%(sten_val_abs)d' % vars()
            flavor_list = control_d['flavor'+suffix]
            if not Core_Util.is_sequence( flavor_list ): # ensure list
                flavor_list = [ flavor_list ]
            depth_list = control_d['depth'+suffix]
            if not Core_Util.is_sequence( depth_list ): # ensure list
                depth_list = [ depth_list ]
            flavor_list = flavor_list[::-1] # reverse
            depth_list = depth_list[::-1] # reverse
            for item, depth in enumerate( depth_list ):
                flavor_val = flavor_list[item]
                rad_val = 1 - depth/pid_d['radius_km']
                if stencil_value == 0:
                    flavor[np.where(((sv>=stencil_value) & (r>=rad_val)))] = flavor_val
                else:
                    flavor[np.where(((sv==stencil_value) & (r>=rad_val)))] = flavor_val

    # build deep layer tracers
    if DEEP_LAYER_TRACERS:
        deep_layer_thickness = control_d['deep_layer_thickness']
        deep_layer_flavor = control_d['deep_layer_flavor']
        radius_inner = pid_d['radius_inner']
        deep_layer_r = radius_inner + deep_layer_thickness/radius_km
        # update flavor array
        flavor[np.where(r<=deep_layer_r)] = deep_layer_flavor

    # write out to single file
    make_dir( trac_dir )
    filename =  '%(trac_dir)s/%(model_name)s.tracer.%(age)sMa' % vars()
    txt = 'writing tracer file= %(filename)s' % vars()
    if verbose: print( now(), txt )
    head = '%(num)s 4' % vars()
    format = '%.6f %.6f %.6f %d'
    out_data = np.column_stack( (theta,phi,r,flavor) )
    np.savetxt( filename, out_data, header=head, fmt=format, comments='' )

    if PLOT_SUMMARY_POSTSCRIPT:
        make_tracer_summary_postscript( master_d )

    # XXX DJB - for testing efficiency
    t1 = time.time() # end time
    runtime = datetime.timedelta(seconds=t1-t0)
    logging.info(f'output_tracer: runtime {runtime}' )
    runtime_per_mil = round((t1-t0) / num * 1E6,3)
    logging.info(f'output_tracer: runtime per million tracers (s) {runtime_per_mil}')

#=====================================================================
#=====================================================================
#=====================================================================
def output_initial_condition( master ):

    '''Output initial condition.'''

    if verbose: print( now(), 'output_initial_condition:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = str( control_d['age'] )
    ic_dir = control_d['ic_dir']
    model_name = control_d['model_name']
    nproc_surf = pid_d['nproc_surf']
    proc_node = pid_d['proc_node']
    temp_by_cap = func_d['temp_by_cap']
    total_proc = pid_d['total_proc']

    # dummy data for vx, vy, vz
    dummy = [[0]*proc_node for ii in range( total_proc )]
    temp_by_proc = Core_Citcom.get_proc_list_from_cap_list( pid_d,
                   temp_by_cap )
    make_dir( ic_dir )
    outname = ic_dir + '/' + model_name + '.velo.#.' + age
    out_data = (dummy, dummy, dummy, temp_by_proc)
    temp_proc_names = Core_Citcom.write_cap_or_proc_list_to_files(
                      pid_d, outname, out_data, 'proc', True )

#=====================================================================
#=====================================================================
#=====================================================================
def output_lith_age( master_d ):

    '''Output CitcomS lithosphere age (Ma) files with no assimilation
       regions set to no_ass_age (if appropriate).'''

    if verbose: print( now(), 'output_lith_age:' )

    # dictionaries
    control_d = master_d['control_d']
    func_d = master_d['func_d']
    pid_d = master_d['pid_d']

    # parameters
    age = control_d['age']
    afile_2 = control_d['afile_2'] # XXX DJB will break if afile_2 missing
    coor_cap_names = func_d['coor_cap_names']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    grd_res = control_d['grd_res']
    IC = control_d['IC']
    lith_age_dir = control_d['lith_age_dir']
    lith_age_min = control_d['lith_age_min']
    model_name = control_d['model_name']
    NO_ASSIM = control_d['NO_ASSIM']
    no_ass_age = control_d['no_ass_age']
    rm_list = func_d['rm_list']
    utbl_age_max = control_d['utbl_age_max']

    # set no assimilation regions to NaN
    # XXX DJB - need and?
    if NO_ASSIM: # and afile_2:
        no_assimilation_regions( master_d, afile_2, grd_res )

    track_file = 'output_lith_age_track.xyz'
    rm_list.append( track_file )
    age_by_cap = []

    for cap_name in coor_cap_names:
        cmd = cap_name + ' -G' + afile_2
        callgmt( 'grdtrack', cmd, '' , '>', track_file )
        age_a = np.loadtxt( track_file, usecols=(2,), unpack=True )
        age_a = age_a.clip( lith_age_min, utbl_age_max )
        # no assimilation regions are defined by NaNs in afile_2
        # do not need to check NO_ASSIM here since NaNs will
        # not exist in afile_2 if NO_ASSIM is False (or no file found)
        age_a[np.isnan( age_a )] = no_ass_age
        age_l = np.around( age_a, decimals=3 ).tolist()
        age_by_cap.append( age_l )

    make_dir( lith_age_dir )
    out_name = lith_age_dir + '/' + model_name + '.lith.dat' + str(age)
    out_name2 = lith_age_dir + '/' + model_name + '.lith.dat' + str(age+1)

    if FULL_SPHERE:
        out_name += '.#' # cap number suffix
        out_name2 += '.#'

    Core_Citcom.write_cap_or_proc_list_to_files( pid_d, 
                               out_name, (age_by_cap,), 'cap', False )

    if IC:
        Core_Citcom.write_cap_or_proc_list_to_files( pid_d, 
                              out_name2, (age_by_cap,), 'cap', False )

#=====================================================================
#=====================================================================
#=====================================================================
def output_regional_bvel( master ):

    '''Export surface velocities for regional models to be read by
       CitcomS.'''

    if verbose: print( now(), 'output_regional_bvel:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = control_d['age']
    bvel_dir = control_d['bvel_dir']
    coor_cap_name = func_d['coor_cap_names'][0]
    IC = control_d['IC']
    rm_list = func_d['rm_list']
    velocity_grid_names = func_d['velocity_grid_names']

    make_dir( bvel_dir )
    filename = bvel_dir + '/bvel.dat' + str(age)
    filename2 = bvel_dir + '/bvel.dat' + str(age+1)

    track_file = 'output_regional_bvel_track.xyz'
    rm_list.append( track_file )

    data_list = []
    for grid in velocity_grid_names:
        cmd = coor_cap_name + ' -G' + grid
        callgmt( 'grdtrack', cmd, '', '>', track_file )
        data = np.loadtxt( track_file, usecols=(2,), unpack=True )
        data_list.append( data.tolist() )

    # ugly, but need to store each list separately by cap number
    # for write_cap_or_proc_list_to_files()
    data1 = [0]
    data1[0] = data_list[0]
    data2 = [0]
    data2[0] = data_list[1]
    bvel_data = ( data1, data2 )

    # set velocity at nodes at edge of domain to zero to avoid
    # problems with pressure convergence
    for data in bvel_data:
        data = Core_Citcom.conform_regional_bvel_sides( pid_d, data )

    Core_Citcom.write_cap_or_proc_list_to_files( pid_d, filename,
        bvel_data, 'cap', False )

    if IC:
        Core_Citcom.write_cap_or_proc_list_to_files( pid_d, filename2,
        bvel_data, 'cap', False )

#=====================================================================
#=====================================================================
#=====================================================================
def output_ivel( master ):

    '''Export descent rate of slabs in the upper mantle as internal
       velocity boundary condition files (one per cap) to be read by
       CitcomS (amended version that supports progressive data
       assimilation).'''

    if verbose: print( now(), 'output_ivel:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = control_d['age']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    IC = control_d['IC']
    ivel_dir = control_d['ivel_dir']
    REGIONAL = pid_d['REGIONAL']
    nproc_surf = pid_d['nproc_surf']

    outname = ivel_dir + '/ivel.dat' + str(age)
    outname2 = ivel_dir + '/ivel.dat' + str(age+1)

    make_ivel_stencil_by_cap( master )

    if REGIONAL:
        # set an additional 2 nodes either side of the boundary to free nodes
        # which should improve convergence of divv/v
        # N.B. last argument is 3 because this works on the FINEST mesh not the
        # COARSEST mesh.  Therefore to capture 2 edge nodes in the coarsest
        # mesh requires 3 edge nodes in the finest mesh (function of `levels')
        Core_Citcom.conform_regional_sides( pid_d,
                                      func_d['ivel_stencil_by_cap'], 0, 3 )
        # must not impose ivels at boundary (stencil of 2 is ignored by
        # CitcomS with data assimilation)
        Core_Citcom.conform_regional_sides( pid_d, 
                                      func_d['ivel_stencil_by_cap'], 2, 1 )
    elif FULL_SPHERE:
        # placeholder
        outname += '.#' # cap number suffix
        outname2 += '.#' # for IC only

    Core_Citcom.conform_top_and_bottom_surfaces( pid_d, 
                                   func_d['ivel_stencil_by_cap'], 2, 2 )

    # number of imposed velocity nodes (duplicates shared nodes for global case)
    stencil_by_cap = func_d['ivel_stencil_by_cap']
    ccount = sum( stencil_by_cap[cc].count( 1 ) for cc in \
        range( nproc_surf ) )
    if verbose: print( now(), 'number of imposed velocity nodes=', ccount )

    # data to output in columns
    ivel_data = (func_d['ivel_velocity_by_cap'], func_d['ivel_stencil_by_cap'])

    make_dir( ivel_dir )
    Core_Citcom.write_cap_or_proc_list_to_files( pid_d, outname, 
                                             ivel_data, 'cap', False )

    # create previous age for IC (required by CitcomS)
    if IC:
        Core_Citcom.write_cap_or_proc_list_to_files( pid_d, outname2,
                                             ivel_data, 'cap', False )

#=====================================================================
#=====================================================================
#=====================================================================
def preprocess_gplates_line_data( master, gplates_xy_filename ):

    logging.info(f'start converting lon to [0,360] and increase resolution of line data in file {gplates_xy_filename}.')
    
    control_d = master['control_d']
    func_d = master['func_d']
    res = control_d['gplates_line_resolution_km'] # in km
    rm_list = func_d['rm_list']

    # GPlates exports lon [-180,180] but CitcomS (and age grids) use
    # lon [0,360].  Convert here and subsequently always use [0,360]
    out_name = gplates_xy_filename.split('/')[-1].rstrip('xy')
    out_name += 'g.xy' # g for GMT global [0,360]
    rm_list.append( out_name )
    in_name = Core_Util.convert_coordinates( gplates_xy_filename, 
        'lon', 'lat', out_name, 'lon', 'lat', 'g' )

    # increase resolution of line data
    out_name = out_name.rstrip('xy') + str(res) + '.xy'
    rm_list.append( out_name )
    # save this out
    out_name1 = Core_Util.increase_resolution_of_xy_line( in_name, res, 
        out_name, True )
    
    return out_name1

#=====================================================================
#=====================================================================
#=====================================================================
def set_global_defaults( arg, pid_d ):

    '''Set global defaults.  These values are not usually adjusted,
       but can also be specified in the input configuration file.'''

    if verbose: print( now(), 'set_global_defaults:' )

    # get flags
    REGIONAL = pid_d['REGIONAL']
    FULL_SPHERE = pid_d['FULL_SPHERE']

    # CitcomS may be hard-coded to assume temperatures [0,1] so let
    # us force this script to do the same
    arg.setdefault( 'temperature_min', 0.0 )
    arg.setdefault( 'temperature_cmb', 1.0 )

    # maximum temperature is used for clipping and therefore must be
    # correct!
    temp_max = arg['temperature_mantle']
    if arg['BUILD_ADIABAT']: temp_max += arg['adiabat_temp_drop']
    if arg['BUILD_LTBL']: temp_max = arg['temperature_cmb']
    arg.setdefault( 'temperature_max', temp_max )

    arg.setdefault('gmt_char', '>')
    arg.setdefault('tension', 0.1)

    # padding for no assimilation stencils
    no_ass_padding = arg.setdefault('no_ass_padding', 0)

    # function of grd_res
    grd_res = arg.setdefault('grd_res', 0.25)
    arg.setdefault('gplates_line_resolution_km', 20)
    # temperature filter width is 55 km
    filter_width = arg.setdefault('filter_width', 110.0*2*grd_res) # km
    # stencil filter width is 110 km
    stencil_filter_width = arg.setdefault('stencil_filter_width', 110.0*4*grd_res) # km
    # degrees per point for the background (1.25)
    spacing_bkg_pts = arg.setdefault('spacing_bkg_pts', 10.0*grd_res)
    # degrees per point for constructing the slab temperature (1/8)
    spacing_slab_pts = arg.setdefault('spacing_slab_pts', 0.25*grd_res)

    # number of pts used to define the cross-section of the slab
    # (temperature and stencil)
    arg.setdefault('N_slab_pts', 121)

    # max depth for construction of the lithosphere temperature grids
    arg.setdefault('lith_depth_gen', 320.0) # km

    # stencil
    arg.setdefault('stencil_background', 0.0)
    arg.setdefault('stencil_max', 1.0)
    arg.setdefault('stencil_min', 0.0)
    arg.setdefault('stencil_depth_max', 350.0) # km
    arg.setdefault('stencil_depth_min', 75.0) # km
    arg.setdefault('stencil_smooth_max', 75.0) # km
    # needs to be a small number > 0
    arg.setdefault('stencil_smooth_min', 0.01) # km
    arg.setdefault('stencil_width', 600.0) # km
    arg.setdefault('stencil_width_smooth', 25.0) # km
    arg.setdefault('flat_slab_stencil_depth', 150.0) # km

    arg.setdefault('velocity_scale', 20.0) # cm/yr per plotting inch

    # full_sphere (global)
    if FULL_SPHERE:
        # some GMT settings for plotting
        arg['B'] = '45'
        arg['J'] = 'H180/5' # inches
        arg['R'] = 'g'

    # regional
    elif REGIONAL:
        lon_min = pid_d['lon_min']
        lon_max = pid_d['lon_max']
        lat_min = pid_d['lat_min']
        lat_max = pid_d['lat_max']
        arg['B'] = '10'
        arg['J'] = 'M3.8' # inches
        arg['R'] = '%f/%f/%f/%f' % (lon_min-grd_res, lon_max+grd_res,
                                    lat_min-grd_res, lat_max+grd_res)

    else:
        print( now(), 'ERROR: nproc_surf must be 1 or 12 in the pid file' )
        sys.exit(1)

    # by default, export data to CitcomS, but maybe this will
    # change eventually, or we'll include more export options
    arg.setdefault('CITCOM_EXPORT', True)

    # flag to either build the horizontal part of flat slabs, or not
    # these are built by default
    arg.setdefault('FLAT_SLAB_RAMP', True)
    
    # allowing the depth of the UM to be different from 660
    arg.setdefault('UM_depth', 660.0) #km

    pid_d.setdefault('tracers_per_element', 10)
    pid_d.setdefault('Solver', 'multigrid')
    pid_d.setdefault('levels', 4)
    pid_d.setdefault('mgunitx', 4)
    pid_d.setdefault('mgunity', 4)
    pid_d.setdefault('mgunitz', 4)

#=====================================================================
#=====================================================================
#=====================================================================
def make_ivel_stencil_by_cap( master ):

    '''Make the velocity stencil.'''

    if verbose: print( now(), 'make_ivel_stencil_by_cap:' )

    t0 = time.time() # start time
    coor_d = master['coor_d']
    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = control_d['age']
    coarse_coor_by_cap = func_d['coarse_coor_by_cap']
    coor_by_cap = func_d['coor_by_cap']
    coord_file = control_d['coord_file']
    DEBUG = control_d['DEBUG']
    depth_km = coor_d['depth_km']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    gmt_char = control_d['gmt_char']
    levels = control_d['levels']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    nodez = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']
    radius_nd = coor_d['radius']
    rm_list = func_d['rm_list']
    roc = control_d['radius_of_curvature']
    ivel_slab_age_xyz = func_d['ivel_slab_age_xyz']
    SYNTHETIC = control_d['SYNTHETIC']
    velocity_grids = func_d['velocity_grid_names']
    vertical_slab_depth = control_d['vertical_slab_depth']

    in_filename = 'sample_point.xy'
    rm_list.append( in_filename )
    out_filename = 'sample_point_velocity.xy'
    rm_list.append( out_filename )

    # need to reverse depth_km for bisect to work correctly
    # determine coarsest multigrid mesh for z nodes    
    coarse_rdepth_km = list( reversed( depth_km ) )[::2**(levels-1)]

    # master KDTree with (duplicated) shared nodes at the edges of caps
    # built based on coarsest multigrid mesh
    coarse_coor_by_cap_flatten = Core_Util.flatten_nested_structure( coarse_coor_by_cap )
    KDTree_all = spatial.KDTree( np.array( coarse_coor_by_cap_flatten ) )

    # initialize stencil and velocity by cap to store the stencil
    # switch; 0 off, 1 on
    stencil_by_cap = []
    velocity_by_cap = []
    KDTree_by_cap = []
    for cc in range( nproc_surf ):
        stencil_by_cap.append( cc )
        velocity_by_cap.append( cc )
        KDTree_by_cap.append( cc )
        stencil_by_cap[cc] = [0 for pp in range( nodex*nodey*nodez )]
        velocity_by_cap[cc] = [ (0,0,0) for pp in range( nodex*nodey*nodez) ]
        # built based on finest multigrid mesh
        KDTree_by_cap[cc] = spatial.KDTree( np.array( coor_by_cap[cc] ) )

    # for debugging
    if DEBUG:
        debug_filename = 'debug_ivel.%(age)s.xy' % vars()
        debug_file = open( debug_filename, 'w' )

    # loop over line data
    for line in open( ivel_slab_age_xyz ):
        # header line
        if line.startswith( gmt_char ):
            polarity = line[2]
            line_segment = line
            line_segments = line.split(' ')
            slab_depth = float(line_segments[1].lstrip('DEPTH=') )
            slab_dip = float(line_segments[2].lstrip('DIP=') )
            slab_dip = np.radians(slab_dip) # to radians
            start_depth = float(line_segments[3].lstrip('START_DEPTH=') )

            sten_depth, sten_smooth = \
                Core_Util.get_stencil_depth_and_smooth( control_d, slab_depth )

            # for cc in [0,0.25,0.5] will apply
            # ivel at 3 depths about the slab center line
            # (1) node just above stencil depth
            # (2) node about 3/4 stencil depth
            # (3) node about 1/2 stencil depth
            depth_list = []
            # At present, we only prescribe one velocity in the
            # slab at the stencil depth (for cc in [0])
            if SYNTHETIC: cc_list = [0,0.25] # 2 points for synthetic
            else: cc_list = [0] # 1 point otherwise

            for cc in cc_list: #[0,0.25,0.5]:
                depth = sten_depth - cc*(sten_depth-start_depth)
                depth_list.append( depth )

            if verbose: print( now(), 'depth_list=', depth_list )
            dist_list = []
            radius_list = []
            znode_list = []

            for depth in depth_list:
                # find depth node just less than depth
                # index for coarsest mesh level
                index = bisect.bisect_left( coarse_rdepth_km, depth )-1
                # now map this back to finest mesh index
                index *= 2**(levels-1)
                # 0 depth returns negative index which crashes this
                # script
                if index < 0: index = 0
                # for nodes at depths just less than target depths
                znode = nodez - index - 1
                # radius of this node
                rad = radius_nd[znode]
                # -------------------
                # if znode <= nodez-1: OLD CONDITION
                # only impose ivbcs below 250 km to keep them clear of the
                # prescribed bvels at the top of the lithosphere
                if rad < 0.96076: # i.e. 250 km
                    znode_list.append( znode )
                    # dist to center line
                    dist = Core_Util.get_slab_center_dist( 
                        depth_km[znode], start_depth, slab_dip, roc,
                        vertical_slab_depth )
                    dist_list.append( dist )
                    radius_list.append( rad )

            # XXX ensure that ivels are only applied when both 252 
            # and 336 km are available
            if SYNTHETIC and len(znode_list) < 2: znode_list = []

            if verbose: print( now(), 'znode_list=', znode_list )
            if verbose: print( now(), 'radius_list=', radius_list )

            # map from surface node to global for each depth
            map = []
            for yy in range( nodey ):
                for xx in range( nodex ):
                    # surface node for this depth
                    # snode = xx + yy*nodex
                    entry = [znode+(xx*nodez)+(yy*nodex*nodez) for znode in znode_list]
                    map.append( tuple(entry) )

            # to store coordinate data for this line segment
            data_list = [] 

        # coordinate line
        else:
            lon, lat, dummy, age = line.split()
            data_list.append( (float(lon), float(lat), float(age)) )
            if len( data_list ) < 3 : continue

            clon, clat, cage = data_list[-1] # current data
            plon, plat, page = data_list[-2] # previous data
            pplon, pplat, ppage = data_list[-3] # previous previous data
            dx = clon - pplon # for approx gradient about previous data
            dy = clat - pplat # for approx gradient about previous data

            # loop over depth znodes
            for nn in range( len(znode_list) ):
                dist = dist_list[nn]
                rad = radius_list[nn]

                print( line_segment )

                # nearest node to find stencil node
                nlon, nlat = Core_Util.get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, dist, polarity )
                npoint = Core_Util.convert_point_to_cartesian( nlon, nlat, rad )

                # ---------------------------------------------------
                # --------- slab unit normals at this depth ---------
                # ---------------------------------------------------

                # normal to subduction zone (at this depth)
                tlon, tlat = Core_Util.get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, -0.5, polarity )
                tpoint = Core_Util.convert_point_to_cartesian( tlon, tlat, rad )
                ulon, ulat = Core_Util.get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, 0, polarity )
                upoint = Core_Util.convert_point_to_cartesian( ulon, ulat, rad )
                vlon, vlat = Core_Util.get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, 0.5, polarity )
                vpoint = Core_Util.convert_point_to_cartesian( vlon, vlat, rad )
                slab_normal = (vpoint-tpoint) / np.linalg.norm( vpoint-tpoint )
                print( 'slab_normal=', slab_normal )

                # radial normal to subduction zone (at this depth)
                wpoint = Core_Util.convert_point_to_cartesian( ulon, ulat, rad-0.005 )
                slab_radial = (upoint-wpoint) / np.linalg.norm( upoint-wpoint )
                print( 'slab_radial', slab_radial )

                # normal parallel to subduction zone (at this depth)
                # method 1:
                # orientation follows the way the line data is stored
                #pppoint = Core_Util.convert_point_to_cartesian( pplon, pplat, rad )
                #cpoint = Core_Util.convert_point_to_cartesian( clon, clat, rad )
                #slab_parallel2 = (cpoint-pppoint) / np.linalg.norm( cpoint-pppoint )
                #print( 'slab_parallel2=', slab_parallel2 )
                # method 2:
                slab_parallel = np.cross( slab_normal, slab_radial )
                print( 'slab_parallel=', slab_parallel )
    
                # ---------------------------------------------------
                # ------------------- velocities --------------------
                # ---------------------------------------------------

                # need to find a point on subducting lithosphere that 
                # is outside of the velocity smoothing region for 
                # regional models, otherwise the extracted values are
                # less than the actual plate motion velcity.
                if SYNTHETIC:
                    tlon, tlat = Core_Util.get_point_normal_to_subduction_zone(
                                     plon, plat, dx, dy, -4, polarity )

                with open( in_filename, 'w' ) as f:
                    f.write( '%(tlon)s %(tlat)s' % vars() )

                components = [] # store v_theta, v_phi, v_radius
                for grid in velocity_grids:
                    Core_Util.find_value_on_line( in_filename, grid, 
                                                        out_filename )
                    data = np.loadtxt( out_filename )[2] 
                    components.append( data )
                components.append( 0 ) # vr always zero
                vtpr = np.array( components )
                print( 'vtpr=', vtpr )
                vxyz = Core_Util.get_cartesian_velocity_for_point(
                                              ulon, ulat, vtpr )
                print( 'vxyz=', vxyz )
                print( 'mod(vxyz)=', np.linalg.norm( vxyz ) )

                #slab_parallel_velocity2 = np.dot( slab_parallel2, vxyz )
                #print( 'slab_parallel_velocity2=', slab_parallel_velocity2 )

                slab_parallel_velocity = np.dot( slab_parallel, vxyz )
                print( 'slab_parallel_velocity=', slab_parallel_velocity )

                subduction_velocity = np.dot( slab_normal, vxyz )
                print( 'subduction_velocity=', subduction_velocity )

                # negative subduction velocity means that there is no
                # convergence (actually means there is divergence).
                #  In this case, do not construct ivbcs and just skip
                # to the next entry in the loop
                if subduction_velocity < 0 : continue

                # ---------------------------------------------------
                # ------------ local xyz coordinate axes ------------
                # ---------------------------------------------------
                # normal at slab location (about dist)
                x1lon, x1lat = Core_Util.get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, dist-0.5, polarity )
                x1 = Core_Util.convert_point_to_cartesian( x1lon, x1lat, rad )
                x2lon, x2lat = Core_Util.get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, dist+0.5, polarity )
                x2 = Core_Util.convert_point_to_cartesian( x2lon, x2lat, rad )

                x3 = Core_Util.convert_point_to_cartesian( nlon, nlat, rad-0.005 )

                hori_norm = (x2-x1) / np.linalg.norm( x2-x1 )
                print( 'hori_norm=', hori_norm )
                vert_norm = (npoint-x3) / np.linalg.norm( npoint-x3 )
                print( 'vert_norm=', vert_norm )
                out_norm = np.cross( hori_norm, vert_norm )
                #out_norm = slab_parallel # TESTING
                print( 'out_norm=', out_norm )

                # ---------------------------------------------------
                # --------- partition velocity between comps --------
                # ---------------------------------------------------
                hori_velo = subduction_velocity * np.cos( slab_dip ) \
                                * hori_norm
                # always negative because into the mantle
                vert_velo = -np.absolute(subduction_velocity) * \
                                np.sin( slab_dip ) * vert_norm
                # 'out' meaning 'out-of-plane' velocity
                #out_velo2 = slab_parallel_velocity2 * out_norm
                #print( 'out_velo2=', out_velo2 )

                out_velo = slab_parallel_velocity * out_norm
                #out_velo *= -1 # XXX HACK - probably should be function of sz polarity
                print( 'out_velo=', out_velo )

                # ---------------------------------------------------
                # --------- map to v_theta, v_phi, v_radius ---------
                # ---------------------------------------------------
                #oper2 = np.concatenate((hori_velo, vert_velo, out_velo2))
                #oper2 = oper2.reshape(3,3).T
                #print( 'oper2=', oper2)
                #vxyz_out2 = np.dot(oper2, np.array( [1,1,1] ) ) # vx, vy, vz
                #vtpr_out2 = Core_Util.get_spherical_velocity_for_point(
                #               npoint, vxyz_out2 )

                #print( 'vtpr_out2=', vtpr_out2 )

                oper = np.concatenate((hori_velo, vert_velo, out_velo))
                oper = oper.reshape(3,3).T
                print( 'oper=', oper)
                vxyz_out = np.dot(oper, np.array( [1,1,1] ) ) # vx, vy, vz
                vtpr_out = Core_Util.get_spherical_velocity_for_point(
                               npoint, vxyz_out )

                print( 'vtpr_out=', vtpr_out )

                # KDTree
                # this determines the closest point in the coarsest
                # multigrid mesh
                # XXX DJB TODO: this cannot handle the prime meridian,
                # which probably means that shared nodes are not captured
                # across 359-0 degrees longitude
                min_dist, min_index = KDTree_all.query( np.array( (nlon, nlat) ) )
                min_point = coarse_coor_by_cap_flatten[ min_index ]

                # now use this point to find closest point in finest
                # mesh for each cap
                # first, check that point in coarse mesh is relatively close
                # to the slab center line.  Maybe try 1 degree initially?
                if min_dist < 1: # one degree criteria (note Cartesian approximation)
                    for cc in range( nproc_surf ):
                        tree = KDTree_by_cap[cc]
                        cap_dist, cap_index = tree.query( np.array( min_point ) )
                        print( cap_dist, cap_index )
                        # set tolerance such that all shared nodes in every
                        # cap are included
                        print('Searching for', min_point)
                        if cap_dist < 0.001: # within 0.001 degrees to catch shared nodes
                            dlon, dlat = coor_by_cap[cc][cap_index]
                            print('Found (%f,%f) in cap %d, index %d' % (dlon, dlat, cc, cap_index))
                            
                            map_index = map[cap_index][nn]
                            stencil_by_cap[cc][ map_index ] = 1
                            velocity_by_cap[cc][ map_index ] = tuple( vtpr_out )

                            if DEBUG:
                                line_out = '%(cc)s %(map_index)s %(dlon)s %(dlat)s %(rad)s %(slab_parallel_velocity)s %(subduction_velocity)s ' % vars()
                                line_out += ' '.join( '%s' % x for x in vtpr_out )
                                print( 'Writing:', line_out )
                                debug_file.write( line_out+'\n' )

    # close debug file
    if DEBUG: debug_file.close()

    # runtime of this function 
    t1 = time.time() # end time
    timefmt = str(datetime.timedelta(seconds=t1-t0))
    print( now(), 'make_ivel_stencil_by_cap: runtime', timefmt )

    # update dictionary
    func_d['ivel_stencil_by_cap'] = stencil_by_cap
    func_d['ivel_velocity_by_cap'] = velocity_by_cap

#=====================================================================
#=====================================================================
#=====================================================================
def make_synthetic_age_grid( master ):

    '''Make regional age grid that is consistent with synthetic data.'''

    if verbose: print( now(), 'make_synthetic_age_grid:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = control_d['age']
    fi_min = pid_d['fi_min']
    fi_trench = control_d['fi_trench']
    grd_res = str(control_d['grd_res'])
    grid_dir = control_d['grid_dir']
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    lat_max = pid_d['lat_max']
    lat_min = pid_d['lat_min']
    lith_age_max = str(control_d['lith_age_max'])
    lith_age_min = str(control_d['lith_age_min'])
    lon_max = pid_d['lon_max']
    lon_min = pid_d['lon_min']
    lon_trench = control_d['lon_trench']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    over_mask_grid = func_d['over_mask_grid']
    overriding_age = control_d['overriding_age']
    plate_velocity_km_Myr = control_d['plate_velocity_km_Myr']
    R = control_d['R']
    radius_km = pid_d['radius_km']
    rm_list = func_d['rm_list']
    sub_mask_grid = func_d['sub_mask_grid']
    tension = str(control_d['tension'])

    dlat = (lat_max - lat_min) / (nodex-1)
    dlon = (lon_max - lon_min) / (nodey-1)

    make_dir( grid_dir )

    age_filename = grid_dir + '/lith_age1_%(age)s.xyz' % vars()
    rm_list.append( age_filename )
    age_file = open( age_filename, 'w' )
    for yy in range(nodey):
        lon1 = lon_min + yy*dlon
        y_km = (lon1 - lon_min)*110.0
        for xx in range(nodex):
            lat1 = lat_min + xx*dlat
            plate_age = y_km / plate_velocity_km_Myr # Ma
            age_file.write( '%g %g %g\n' % (lon1, lat1, plate_age) )
    age_file.close()

    # make grid
    median_xyz = age_filename.rstrip('xyz') + 'median.xyz'
    rm_list.append( median_xyz )
    afile_1 = age_filename.rstrip('xyz') + 'grd'
    if not KEEP_GRIDS: rm_list.append( afile_1 )
    cmd = age_filename + ' -I' + grd_res + ' -R' + R
    callgmt( 'blockmedian', cmd, '', '>', median_xyz )
    cmd = median_xyz + ' -I' + grd_res + ' -R' + R + ' -T' + tension
    callgmt( 'surface', cmd, '', '', ' -G' + afile_1 )

    # age file 2 with continental thermal age
    afile_2 = afile_1.rstrip( '1_%(age)s.grd' % vars() )
    afile_2 += '2_%(age)s.grd' % vars()
    if not KEEP_GRIDS: rm_list.append( afile_2 )
    # subducting plate contribution
    args = '%(afile_1)s %(sub_mask_grid)s MUL' % vars()
    callgmt( 'grdmath', args, '', '=', afile_2 )
    # over-riding plate contribution
    args = '%(overriding_age)s %(over_mask_grid)s MUL %(afile_2)s ADD' % vars()
    callgmt( 'grdmath', args, '', '=', afile_2 )

    # afile_3 no longer exists for the DATA case
    # afile_3 for plotting only
    #afile_3 = afile_1.rstrip( '1_%(age)s.grd' % vars() )
    #afile_3 += '3_%(age)s.grd' % vars()
    #if not KEEP_GRIDS: rm_list.append( afile_3 )
    #args = '%(afile_2)s %(overriding_age)s NAN' % vars()
    #callgmt( 'grdmath', args, '', '=', afile_3 )

    # update control_d dictionary
    control_d['afile_1'] = afile_1
    control_d['afile_2'] = afile_2
    #control_d['afile_3'] = afile_3

#=====================================================================
#=====================================================================
#=====================================================================
def make_synthetic_surface_velocity_grids( master ):

    '''Make regional surface velocity grids from synthetic data to
       make ivel files.'''

    if verbose: print( now(), 'make_synthetic_surface_velocity_grids:' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    KEEP_GRIDS = control_d['KEEP_GRIDS']
    afile_2 = control_d['afile_2']
    age = control_d['age']
    grd_res = str( control_d['grd_res'] )
    grid_dir = control_d['grid_dir']
    lat_max = pid_d['lat_max']
    lat_min = pid_d['lat_min']
    lon_min = pid_d['lon_min']
    lon_max = pid_d['lon_max']
    zero_nodes = control_d['no_of_edge_nodes_to_zero']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    over_mask_grid = func_d['over_mask_grid']
    plate_velocity = control_d['plate_velocity'] # cm/yr
    # these quantities only define the direction of the plate velocity
    # and are non-dimensional
    plate_velocity_t = control_d['plate_velocity_theta']
    plate_velocity_p = control_d['plate_velocity_phi']
    rollback_start_age = control_d['rollback_start_age']
    sub_mask_grid = func_d['sub_mask_grid']
    velocity_scale = control_d['velocity_scale']
    velocity_smooth = control_d['velocity_smooth'] # km
    R = control_d['R']
    rm_list = func_d['rm_list']

    velo_names = [ 'theta', 'phi' ]

    if age < rollback_start_age:
        over_velocity_p = -control_d['rollback_cm_yr'] # cm/yr
    else:
        over_velocity_p = 0.0 # stationary over-riding plate
    print( 'over_velocity_p=', over_velocity_p )

    # subducting and over-riding plate velocities
    sub_velo = np.array( [plate_velocity_t, plate_velocity_p] )
    sub_velo = sub_velo / np.linalg.norm( sub_velo ) # unit normal
    sub_velo = plate_velocity * sub_velo # theta and phi components
    over_velo = np.array( [0, over_velocity_p] )

    # make blending grid to smooth velocity at boundaries to zero
    blend_filename = grid_dir + '/blend_velocity.xyz'
    rm_list.append( blend_filename )
    blend_file = open( blend_filename, 'w' )
    dlat = (lat_max - lat_min) / (nodex-1)
    dlon = (lon_max - lon_min) / (nodey-1)
    out = 1
    for yy in range(nodey):
        lon1 = lon_min + yy*dlon
        for xx in range(nodex):
            lat1 = lat_min + xx*dlat
            if yy <= (zero_nodes - 1) or yy >= nodey - (zero_nodes+1):
                out = 0
            if xx <= (zero_nodes - 1) or xx >= nodex - zero_nodes:
                out = 0
            blend_file.write( '%g %g %g\n' % (lon1, lat1, out) )
            out = 1
    blend_file.close()

    # make blending grid
    median_xyz = blend_filename.rstrip('xyz') + 'median.xyz'
    rm_list.append( median_xyz )
    blend_grid = blend_filename.rstrip('xyz') + 'grd'
    rm_list.append( blend_grid )
    cmd = blend_filename + ' -I' + grd_res + ' -R' + R
    callgmt( 'blockmedian', cmd, '', '>', median_xyz )
    cmd = median_xyz + ' -I' + grd_res + ' -R' + R + ' -Ll0.0 -Lu1.0'
    callgmt( 'surface', cmd, '', '', ' -G' + blend_grid )

    # make grids
    # sample afile_2 for simplicity
    grid_names = []
    for nn, name in enumerate( velo_names ):
        # velocity components of subducting plate
        subvelo = sub_velo[nn]
        overvelo = over_velo[nn]
        velo_grid = grid_dir + '/velocity_' + name + '.%(age)sMa.grd' % vars()
        grid_names.append( velo_grid )
        if not KEEP_GRIDS: rm_list.append( velo_grid )
        args = 'cp %(afile_2)s %(velo_grid)s' % vars()
        subprocess.call( args, shell=True )
        # subducting plate contribution
        args = '%(velo_grid)s 0 MUL %(subvelo)s %(sub_mask_grid)s MUL ADD' % vars()
        callgmt( 'grdmath', args, '', '=', velo_grid )
        # over-riding plate contribution
        args = '%(overvelo)s %(over_mask_grid)s MUL %(velo_grid)s ADD' % vars()
        callgmt( 'grdmath', args, '', '=', velo_grid )
        # multiply by blending grid to smooth outer edges
        args = '%(velo_grid)s %(blend_grid)s MUL' % vars()
        callgmt( 'grdmath', args, '', '=', velo_grid )
        # smooth velocity gradients
        args = '%(velo_grid)s -D2 -Fg%(velocity_smooth)s' % vars()
        callgmt( 'grdfilter', args, '', '', '-G' + velo_grid )

    func_d['velocity_grid_names'] = grid_names

    # create GMT 4-column file containing the vector information
    # for plotting surface velocities on the plate history model
    # in the summary postscript
    vtp_grid_files_l = func_d['velocity_grid_names']
    out_file_name = 'velocity_map.xyz'
    rm_list.append( out_file_name )
    # XXX this part is (unfortunately) hard-coded for the regional
    # domain in Bower et al. (2014), PEPI
    # if you try and use a different regional domain, this can produce
    # errors of the sort:
    #     lon = velo_data[0][:,0]
    #     IndexError: too many indices
    # because no data is found in the grdtrack, no data is input into 
    # the velo_data array.  This can probably be cleaned up at some point.
    velocity_mesh = 'velocity_map_mesh.xy'
    rm_list.append( velocity_mesh )
    x = np.arange(-13,13,8) # latitude
    y = np.arange(1,57,6) # longitude

    coordinates = np.vstack(np.meshgrid(y, x)).reshape(2,-1).T
    np.savetxt( velocity_mesh, coordinates, fmt='%f %f' )

    Core_Util.make_map_velocity( master, velocity_mesh, \
        velocity_scale, vtp_grid_files_l, out_file_name )

    func_d['velocity_map'] = out_file_name

#=====================================================================
#=====================================================================
#=====================================================================
def make_synthetic_subduction_file( master ):

    '''Make regional synthetic subduction data file.'''

    if verbose: print( now(), 'make_synthetic_subduction_file:')

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    age = control_d['age']
    fi_max = pid_d['fi_max']
    fi_min = pid_d['fi_min']
    fi_trench = control_d['fi_trench']
    grd_res = control_d['grd_res']
    lat_max = pid_d['lat_max']
    lat_min = pid_d['lat_min']
    lon_max = pid_d['lon_max']
    lon_min = pid_d['lon_min']
    plate_velocity = control_d['plate_velocity']
    plate_velocity_km_Myr = control_d.setdefault('plate_velocity_km_Myr', plate_velocity*10.0)
    radius_km = pid_d['radius_km']
    rm_list = func_d['rm_list']
    rollback_cm_yr = control_d['rollback_cm_yr']
    rollback_vel = control_d.setdefault('rollback_vel',
                               rollback_cm_yr*10.0/radius_km) # rad
    rollback_start_age = control_d['rollback_start_age']
    slab_dip = control_d['default_slab_dip']
    start_depth = 0
    subduction_zone_age = control_d['subduction_zone_age']
    TRENCH_CURVING = control_d['TRENCH_CURVING']

    # rollback (time-dependent fi_trench) begins after rollback_start_age
    if rollback_vel > 0 and age < rollback_start_age:
        fi_trench -= (rollback_start_age-age)*rollback_vel

    if (fi_trench < fi_min) or (fi_trench > fi_max):
        print( now(), 'ERROR: fi_trench (%f) out of bounds' % fi_trench)
        sys.exit(1)
    lon_trench = np.degrees( fi_trench )
    control_d['lon_trench'] = lon_trench # needed for age grids

    slab_filename = 'synthetic_subduction_boundaries_%(age)0.2fMa.g.head.10.xy' % vars()
    rm_list.append( slab_filename )
    slab_file = open( slab_filename, 'w' ) 

    # TESTING: make two subduction zones
    lon_trench_list = [ lon_trench ] #, 22.5]
    for nn, polarity in enumerate( ['sR'] ): #, 'sL'] ):
        lon_trench = lon_trench_list[nn]
        header_line = '>' + polarity
        header_line += ' # subductionZoneAge: %(subduction_zone_age)f\n' % vars()
        slab_file.write( header_line )

        lat1 = lat_min - grd_res
        #lat1 = lat_max + grd_res # TESTING
        while lat1 <= lat_max + grd_res:
        #while lat1 >= lat_min - grd_res: # TESTING
            if TRENCH_CURVING:
                curving_trench_lat = control_d['curving_trench_lat']
                if lat1 > curving_trench_lat:
                    rr = 14.5742 - curving_trench_lat
                    dxx = lat1 - curving_trench_lat
                    yy = rr*np.cos(np.arcsin(dxx/rr))
                    dyy = rr-yy
                    lon_trench_lat = lon_trench - dyy
                else:
                    lon_trench_lat = lon_trench
            else: lon_trench_lat = lon_trench
            out_line = '%g %g 0\n' % ( lon_trench_lat, lat1 )
            slab_file.write( out_line )
            lat1 += grd_res
            #lat1 -= grd_res # TESTING
    slab_file.close()

    # XXX construct a circular subduction zone to test ivels
    # N.B. also need to change plotting for correct polarity
    #header_line = '>' + 'sL'
    #header_line += ' # subductionZoneAge: %(subduction_zone_age)f\n' % vars()
    #slab_file.write( header_line )
    #rr = 7.0 # degrees
    #for theta in range(0,361):
    #    xx = rr*np.cos(np.radians(theta))
    #    yy = rr*np.sin(np.radians(theta))
    #    yy += 29
    #    out_line = '%g %g 0\n' % (yy,xx)
    #    slab_file.write( out_line )
    #slab_file.close()

    # pass values using function dictionary
    func_d['flat_age_xyz'] = None
    func_d['slab_age_xyz'] = slab_filename

    # necessary for the subduction zone to plot in the summary ps
    control_d['sub_file_right'] = slab_filename

#=====================================================================
#=====================================================================
#=====================================================================
def make_synthetic_masks( master_d ):

    '''make GMT masks for subducting and over-riding plate to use in
       the construction of the age and surface velocity grids.'''

    control_d = master_d['control_d']
    func_d = master_d['func_d']
    pid_d = master_d['pid_d']
    grd_res = control_d['grd_res']
    grid_dir = control_d['grid_dir']
    lat_max = pid_d['lat_max']
    lat_min = pid_d['lat_min']
    rm_list = func_d['rm_list']
    R = control_d['R']
    slab_age_xyz = func_d['slab_age_xyz']

    mask_xy = slab_age_xyz.rstrip('xy') + 'mask.xy'
    rm_list.append( mask_xy )
    cmd = 'cp %(slab_age_xyz)s %(mask_xy)s' % vars()
    subprocess.call( cmd, shell=True )

    # close off the polygon
    # XXX only works for sR
    mask_file = open( mask_xy, 'a' )
    # third and fourth columns contain dummy values
    lat_max1 = lat_max + 0.25 
    lat_min1 = lat_min - 0.25 
    # XXX comment out next two lines for testing with circular
    # subduction zone
    mask_file.write( '0 %(lat_max1)s 0 0\n' % vars() )
    mask_file.write( '0 %(lat_min1)s 0 0\n' % vars() )
    mask_file.close()

    # make mask for subducting plate
    sub_mask_grid = grid_dir + '/subducting_mask.grd'
    rm_list.append( sub_mask_grid )
    cmd = mask_xy + ' -R%(R)s -I%(grd_res)s -m' % vars()
    callgmt( 'grdmask', cmd, '', '', '-G' + sub_mask_grid )

    # make mask for over-riding plate
    over_mask_grid = grid_dir + '/overriding_mask.grd'
    rm_list.append( over_mask_grid )
    cmd += ' -N1/0/0'
    callgmt( 'grdmask', cmd, '', '', '-G' + over_mask_grid )

    # update dictionary
    func_d['sub_mask_grid'] = sub_mask_grid
    func_d['over_mask_grid'] = over_mask_grid

#=====================================================================
#=====================================================================
#=====================================================================
def get_synthetic_data( master_d ):

    '''Produce synthetic data files for regional trial models.'''

    logging.info('start get_synthetic_data')

    control_d = master_d['control_d']
    func_d = master_d['func_d']

    # make synthetic subduction file
    make_synthetic_subduction_file( master_d )

    # get header data
    slab_age_xyz = master_d['func_d']['slab_age_xyz']
    Core_Util.get_slab_data( master_d['control_d'], slab_age_xyz, slab_age_xyz )

    # make masks
    make_synthetic_masks( master_d )

    # make afile_1 and afile_2
    make_synthetic_age_grid( master_d )

    # make surface velocity grids
    make_synthetic_surface_velocity_grids( master_d )

    # get thermal ages from modified age grid (without mask)
    in_file = func_d['slab_age_xyz']
    out_file = in_file.rstrip('xy') + 'age.xy'
    func_d['slab_age_xyz'] = Core_Util.find_value_on_line( in_file,
        control_d['afile_1'], out_file )

    # for ivels
    func_d['ivel_slab_age_xyz'] = func_d['slab_age_xyz']

#=====================================================================
#=====================================================================
#=====================================================================
def set_verbose( control_d ):

    '''Set verbose for this script and all Core modules that
       are imported.'''

    verbose_list = [Core_Citcom.verbose, 
                    Core_GMT.verbose, Core_Util.verbose, verbose]

    if control_d['VERBOSE']: val = True
    else: val = False
    for entry in verbose_list: entry = val

#=====================================================================
#=====================================================================
#=====================================================================
def write_coordinates_by_cap( master ):

    '''Export CitcomS coordinates for each cap.  If ivels are to be
       exported, additionally write out the coordinates of a coarsened
       mesh (coarsened by a factor of 2**(levels-1)).'''

    logging.info('start write_coordinates_by_cap' )

    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    coord_file = control_d['coord_file']
    rm_list = func_d['rm_list']
    levels = control_d['levels']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    nproc_surf = pid_d['nproc_surf']
    OUTPUT_IVEL = control_d['OUTPUT_IVEL']

    coor_by_cap =  Core_Citcom.read_citcom_surface_coor( pid_d, 
                                                          coord_file )
    outname = 'coord.cap.#'
    coor_cap_names = Core_Citcom.write_cap_or_proc_list_to_files( pid_d,
                               outname, (coor_by_cap,), 'cap', False )

    rm_list.append( coor_cap_names )

    # update dictionary
    func_d['coor_cap_names'] = coor_cap_names
    func_d['coor_by_cap'] = coor_by_cap

    if OUTPUT_IVEL:
        # determine surface coordinates for coarsest multigrid mesh
        nox1 = int((nodex-1)/(2**(levels-1))+1)
        noy1 = int((nodey-1)/(2**(levels-1))+1)
        coarse_coor_by_cap = []
        for cc in range( nproc_surf ):
            coarse_coor_by_cap.append([])
            for jj in range( noy1 ):
                for ii in range( nox1) :
                    nodel = ii + nox1*jj
                    node = ii*2**(levels-1)
                    node += jj*nodex*2**(levels-1)
                    coarse_coor_by_cap[cc].append( coor_by_cap[cc][node] )

        coarse_outname = 'coarse.coord.cap.#'
        coarse_coor_cap_names = Core_Citcom.write_cap_or_proc_list_to_files( pid_d, coarse_outname, (coarse_coor_by_cap,), 'cap', False )

        rm_list.append( coarse_coor_cap_names )

        # update dictionary
        func_d['coarse_coor_cap_names'] = coarse_coor_cap_names
        func_d['coarse_coor_by_cap'] = coarse_coor_by_cap


#====================================================================
#====================================================================
#====================================================================
def isSerial(control_d):
    result = False;
    if(control_d['job']=='smp'):
        if(control_d['nproc']==1): result = True;
        elif((control_d['nproc']==-1) and \
             (int(multiprocessing.cpu_count())==1)): result = True;
        else: result = False;
    elif (control_d['job']=='cluster'):
        result = False;
    elif (control_d['job']=='raijin'):
        result = False;
    elif (control_d['job']=='baloo'):
        result = False;
    else:
        print ("Illegal option for 'job', check config file. Aborting..")
        sys.exit(0)
    #end if

    return result
#end function


#=====================================================================
#=====================================================================
#=====================================================================

if __name__ == "__main__":

    t0 = time.time() # start time
    main()
    t1 = time.time() # end time
    timefmt = str(datetime.timedelta(seconds=t1-t0))
    print( now(), 'main: runtime', timefmt )

#=====================================================================
#=====================================================================
#=====================================================================
