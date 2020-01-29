#!/usr/bin/env python
#=====================================================================
#                 Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''Core_Citcom.py holds a set of general purpose functions for working with Citcoms data files.

This module has functions to read and parse various file types associated 
with CitComS and closely related data sets.

Most functions take a dictionary of arguments to pass and adjust parameters.
'''
#=====================================================================
#=====================================================================
import os, re, string, sys, traceback, glob, subprocess, logging
import numpy as np

# Caltech geodynamic framework modules:

import Core_Util
from Core_Util import now

import Core_GMT

#=====================================================================
#=====================================================================
# Global Variables for this module 
verbose = True

#=====================================================================
# The field_to_file_map is a dictionary of dictionaries holding the mapping 
# from a Citcom field name to various meta-data about where the field appears 
# in Citcom output files.
#
# Key 'file' gives the filename component, e.g. 'data/0/gld29.velo.0.5400'
# Key 'column' gives a zero-based index of the data colum (e.g. 3 = fourth col)
# Key 'header_count' gives the number of header lines in the output file
#
field_to_file_map = {

    # coordinate data; NOTE: column is a placeholder 
    'coord' : { 'file' : 'coord', 'column' : 0, 'header_count' : 1 }, # coordinate data

    # fields in .velo. files
    'vx'         : { 'file' : 'velo', 'column' : 0, 'header_count' : 2 }, # velocity x  
    'vy'         : { 'file' : 'velo', 'column' : 1, 'header_count' : 2 }, # velocity y  
    'vz'         : { 'file' : 'velo', 'column' : 2, 'header_count' : 2 }, # velocity z  
    'temp'       : { 'file' : 'velo', 'column' : 3, 'header_count' : 2 }, # temperature 

    # divergence (divv) - DJB
    # can only be output using CitcomS assimilation version of the code
    # N.B., the divergence is not normalized!  You need to reprocess
    # using |divv/vmag| to obtain a quantity similar to the CitcomS
    # log file, where vmag is the magnitude of the total velocity vector.

    # FIXME : check the header line count for these three fiields; Need example files 
    'divv'       : { 'file' : 'divv',       'column' : 0, 'header_count' : 2 }, # divergence
    'sten_temp'  : { 'file' : 'sten_temp',  'column' : 0, 'header_count' : 2 }, # temperature stencil
    'pressure'   : { 'file' : 'pressure',   'column' : 0, 'header_count' : 2 }, # pressure

    # .visc. data files
    'visc'       : { 'file' : 'visc',       'column' : 0, 'header_count' : 1 }, # visc 

    # .comp_nd. data files
    'comp_nd'    : { 'file' : 'comp_nd',    'column' : 0, 'header_count' : 2 }, # composition
    'comp_nd1'   : { 'file' : 'comp_nd',    'column' : 1, 'header_count' : 2 }, # composition
    'comp_nd2'   : { 'file' : 'comp_nd',    'column' : 2, 'header_count' : 2 }, # composition
    'comp_nd3'   : { 'file' : 'comp_nd',    'column' : 3, 'header_count' : 2 }, # composition
    'comp_nd4'   : { 'file' : 'comp_nd',    'column' : 4, 'header_count' : 2 }, # composition

    # tracer density (feature added by Ting to assimilation output)
    'tracer_dens': { 'file': 'tracer_dens', 'column' : 0, 'header_count' : 1 },

    # .surf. data files
    'surf_topography' : { 'file' : 'surf', 'column' : 0, 'header_count' : 1 }, 
    'surf_heat_flux'  : { 'file' : 'surf', 'column' : 1, 'header_count' : 1 }, 
    'surf_vel_colat'  : { 'file' : 'surf', 'column' : 2, 'header_count' : 1 }, 
    'surf_vel_lon'    : { 'file' : 'surf', 'column' : 3, 'header_count' : 1 }, 

    # .botm. data files
    'botm_topography' : { 'file' : 'botm', 'column' : 0, 'header_count' : 1 }, 
    'botm_heat_flux'  : { 'file' : 'botm', 'column' : 1, 'header_count' : 1 }, 
    'botm_vel_colat'  : { 'file' : 'botm', 'column' : 2, 'header_count' : 1 }, 
    'botm_vel_lon'    : { 'file' : 'botm', 'column' : 3, 'header_count' : 1 }, 

    # gplates data  files
    'gplates_vx' : { 'file' : 'bvel', 'column' : 0 }, # theta (colat) velo
    'gplates_vy' : { 'file' : 'bvel', 'column' : 1 }, # phi (lon) velo
    # NOTE: this is a placeholder for consistancy; specific programs will read both columns and compute total velocity
    'gplates_vmag' : { 'file' : 'bvel', 'column' : 0 }, 
}

#=====================================================================
# The field_to_dimensional_map is a dictionary of dictionaries holding the 
# mapping from a Citcom field name to the coefficients and constants used 
# to compute scaled dimensional data from raw non-dimensional values 
# in Citcom output files.
#
# The general form of the computation is y = m * x + c
# where x is the raw non-dimensional value, y is the computed dimensional value, 
# and the coefficent m, and constant c.
# 
# These values are place holders.  
# Actual values will be computed based on a given pid file 
# in the function populate_field_to_dimensional_map_from_pid() 
#
# The key 'units' has the corresponding SI units for the field

# FIXME: check with Dan and Nico and the citcoms docs about these units 

field_to_dimensional_map = {

    # FIXME: scaling of coordinates may not fit the general form?
    'coord'      : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # coordinate data

    # fields in .velo. files
    'vx'         : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, # velocity x  
    'vy'         : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, # velocity y  
    'vz'         : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, # velocity z  
    'temp'       : { 'coef' : 1, 'const' : 0, 'units' : 'Kelvins' }, # temperature 

    # fields in mixed files; see above
    'divv'       : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # divergence
    'sten_temp'  : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # temperature stencil
    'pressure'   : { 'coef' : 1, 'const' : 0, 'units' : 'Pa' }, # pressure

    # .visc. data files
    # DJB FIXME: scaling of viscosity also does not fit the general form
    'visc'       : { 'coef' : 1, 'const' : 0, 'units' : 'Exponent' }, # visc 

    # .comp_nd. data files

    'comp_nd'    : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # composition
    'comp_nd1'    : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # composition
    'comp_nd2'    : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # composition
    'comp_nd3'    : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # composition
    'comp_nd4'    : { 'coef' : 1, 'const' : 0, 'units' : 'none' }, # composition


    # tracer density (feature added by Ting to assimilation output)
    'tracer_dens': { 'coef' : 1, 'const' : 0, 'units' : 'none' },

    # .surf. data files
    # metres for topography, but we could use km instead
    'surf_topography' : { 'coef' : 1, 'const' : 0, 'units' : 'metres' }, 
    'surf_heat_flux'  : { 'coef' : 1, 'const' : 0, 'units' : 'milliW / m^2' }, 
    'surf_vel_colat'  : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, 
    'surf_vel_lon'    : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, 

    # .botm. data files
    'botm_topography' : { 'coef' : 1, 'const' : 0, 'units' : 'metres' }, 
    'botm_heat_flux'  : { 'coef' : 1, 'const' : 0, 'units' : 'milliW / m^2' }, 
    'botm_vel_colat'  : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, 
    'botm_vel_lon'    : { 'coef' : 1, 'const' : 0, 'units' : 'cm/yr' }, 
}


#=====================================================================
# The next two global variables are dictionaries that 
# hold specific known CitcomS input parameters to modify 
# when copying an original master run input .cfg to a new restart input .cfg
#
# Each key has the Prefix of the particular CitcomS [ ] section.
# Each value is a replacement value for the new .cfg 
# The special value of 'COMMENT' will comment out that entry in the new restart input .cfg 
# The special value of 'DELETE' will omit that value in the new restart input .cfg 
# The sepcial value of 'RS_TIMESTEP' will be replaced by the specific time step from original run to restart from

# NOTE: not all values are listed here.
# Some parameter values are set autoamtically by restart_citcoms.py 
# depending on the age / timestep value of the restart, and,
# other parameter values are left as-is from the original input .cfg 

# Total Topography Restarts  ( also called 'checkpoint' restarts)
total_topography_restart_params = {
    'CitcomS.steps' : 'RS_TIMESTEP+2',

    'CitcomS.controller.monitoringFrequency' : 1,
    'CitcomS.controller.checkpointFrequency' : 1,

    'CitcomS.solver.output.output_optional'  : 'surf,geoid,botm',
    'CitcomS.solver.output.self_gravitation' : 1,
    'CitcomS.solver.output.use_cbf_topo'     : 1,
     
    'CitcomS.solver.ic.restart'              : 'on',
    'CitcomS.solver.ic.solution_cycles_init' : 'RS_TIMESTEP',
    'CitcomS.solver.ic.zero_elapsed_time'    : 'off',
    'CitcomS.solver.ic.tic_method'           : 'COMMENT',
     
    'CitcomS.solver.tracer.tracer_file' : 'COMMENT',
     
    'CitcomS.solver.bc.topvbxval' : 0.0,
    'CitcomS.solver.bc.topvbyval' : 0.0,
     
    'CitcomS.solver.param.file_vbcs'      : 0,
    'CitcomS.solver.param.file_bcs'       : 'DELETE',
    'CitcomS.solver.param.start_age'      : 'COMMENT',
    'CitcomS.solver.param.vel_bound_file' : 'COMMENT',
}

# Dynamic Topogrphy Restart parameters   ( also called 'nolith' restarts)
dynamic_topography_restart_params = {
    'CitcomS.steps' : 0,

    'CitcomS.controller.monitoringFrequency' : 1,
    'CitcomS.controller.checkpointFrequency' : 1,

    'CitcomS.solver.stokes_flow_only' : 'on',

    'CitcomS.solver.output.output_optional' : 'surf,geoid,botm',
    'CitcomS.solver.output.use_cbf_topo' : 1,
    'CitcomS.solver.output.self_gravitation' : 1,

    'CitcomS.solver.ic.solution_cycles_init' : 'RS_TIMESTEP',

    'CitcomS.solver.tracer.tracer' : 'off',
    'CitcomS.solver.tracer.chemical_buoyancy' : 'off',
    'CitcomS.solver.tracer.tracer_file' : 'DELETE',
    'CitcomS.solver.tracer.tracer_flavors' : 'DELETE',
    'CitcomS.solver.tracer.z_interface' : 'DELETE',
    'CitcomS.solver.tracer.chemical_buoyancy' : 'DELETE',
    'CitcomS.solver.tracer.buoy_type' : 'DELETE',
    'CitcomS.solver.tracer.buoyancy_ratio' : 'DELETE',
    'CitcomS.solver.tracer.regular_grid_deltheta' : 'DELETE',
    'CitcomS.solver.tracer.regular_grid_delphi' : 'DELETE',
    
    'CitcomS.solver.bc.topvbxval' : 0.0,
    'CitcomS.solver.bc.topvbyval' : 0.0,
     
    'CitcomS.solver.param.file_bcs' : 'DELETE',
    'CitcomS.solver.param.file_vbcs' : 0,
    'CitcomS.solver.param.start_age' : 'RS_AGE',
    'CitcomS.solver.param.reset_startage' : 'off',
    'CitcomS.solver.param.vel_bound_file' : 'DELETE',
    'CitcomS.solver.param.lith_age' : 'off',
    'CitcomS.solver.param.lith_age_time' : 0,
    'CitcomS.solver.param.slab_assim' : 0,

    'CitcomS.solver.visc.CDEPV' : 'off',
}

#=====================================================================
#=====================================================================
#=====================================================================
def conform_regional_bvel_sides( arg, cap_list ):
    '''Description.'''

    if verbose: print( Core_Util.now(), 'conform_regional_bvel_sides:' )

    only_cap = cap_list[0]

    nodex = arg['nodex']
    nodey = arg['nodey']

    for jj in range( nodey ):
        for ii in range( nodex ):
            nodeg = ii + nodex*jj
            if jj == 0 or jj == nodey-1 or ii == 0 or ii == nodex-1:
                only_cap[nodeg] = 0

    return cap_list

#=====================================================================
#=====================================================================
#=====================================================================
def conform_regional_sides( arg, cap_list, side_val, num ):

    '''Set the edge nodes of a regional cap to side_val.'''

    if verbose: print( Core_Util.now(), 'conform_regional_sides:')

    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']

    only_cap = cap_list[0]
    for jj in range( nodey ):
        for ii in range( nodex ):
            for kk in range( nodez ):
                nodeg = kk + ii*nodez + jj*nodez*nodex
                if jj in range(num) or jj in range(nodey-num,nodey) \
                  or ii in range(num) or ii in range(nodex-num,nodex):
                #if jj == 0 or jj == nodey-1 or ii == 0 or ii == nodex-1:
                    only_cap[nodeg] = side_val

    return cap_list

#=====================================================================
#=====================================================================
#=====================================================================
def conform_top_and_bottom_surfaces( arg, cap_list, top_val, bot_val ):

    '''Set the top and bottom surface nodes to top_val and bot_val,
       respectively, for a cap_list.'''

    if verbose: print( Core_Util.now(), 'conform_top_and_bottom_surfaces:')

    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']
    nproc_surf = arg['nproc_surf']

    for cc in range( nproc_surf ):
        for jj in range( nodey ):
            for ii in range( nodex ):
                bot_node = ii*nodez + jj*nodez*nodex
                top_node = (nodez-1) + bot_node
                cap_list[cc][bot_node] = bot_val
                cap_list[cc][top_node] = top_val

    return cap_list

#=====================================================================
#=====================================================================
#=====================================================================
def define_cap_or_proc_names( arg, filename, cap_or_proc ):

    '''Define the naming convention for cap or processor files,
       either to read or write data.  Replace cap or proc number
       with `#'s where %0(count #)d determines the format string for
       the cap or proc number.  Also determines possible header 
       lines.'''

    logging.info('start defining cap or proc names' )

    # parameters for proc
    if cap_or_proc == 'proc':
        loop_over = arg['total_proc']
        proc_node = arg['proc_node']
        # header 1 is time-dependent, so this is only a dummy
        # header 1 must NOT exactly match any header entry!
        # (hence 00 instead of just 0)
        header1 = '00 %(proc_node)s 0.00000e+00' % vars()
        header2 = '  1'
        header2 += '%s' % str(proc_node).rjust(8) # *.coord.* only
        header_list = ( header1, header2 )

    # parameters for cap
    elif cap_or_proc == 'cap':
        loop_over = arg['nproc_surf']
        nodex = arg['nodex']
        nodey = arg['nodey']
        nodez = arg['nodez']
        header1 = '%(nodex)s x %(nodey)s x %(nodez)s' % vars() # all nodes
        header2 = '%(nodex)s x %(nodey)s' % vars() # surface node only
        header_list = ( header1, header2 )
    # parameters unknown
    else:
        print( Core_Util.now(), 'ERROR: unknown if cap or proc' )
        sys.exit(1)

    file_names = []
    for nn in range( loop_over ):
        for pp in range(4,0,-1):
            filename = filename.replace( 
                                    '#'*pp, '%(nn)0' + str(pp) + 'd' )
        file_names.append( filename % vars() )

    return file_names, header_list

#=====================================================================
#=====================================================================
#=====================================================================
def derive_extra_citcom_parameters( arg ):

    '''Derive additional quantities from CitcomS pid file that are
       required for the geodynamic framework.'''

    logging.info('start deriving extra citcom parameters' )

    ndict = {}

    # for looping procedures
    ndict['nx'] = int((arg['nodex']-1)/arg['nprocx']+1) # x nodes per proc
    ndict['ny'] = int((arg['nodey']-1)/arg['nprocy']+1) # y nodes per proc
    ndict['nz'] = int((arg['nodez']-1)/arg['nprocz']+1) # z nodes per proc
    ndict['cap_proc'] = arg['nprocx']*arg['nprocy']*arg['nprocz'] # processors
    ndict['cap_node'] = arg['nodex']*arg['nodey']*arg['nodez'] # nodes per cap
    ndict['proc_node'] = ndict['nx']*ndict['ny']*ndict['nz'] # nodes per proc
    ndict['total_proc'] = ndict['cap_proc'] * arg['nproc_surf'] # total proc
    ndict['total_node'] = ndict['cap_node'] * arg['nproc_surf'] # total node

    ndict['radius_km'] = arg['radius']*1E-3 # radius of Earth in km

    # total temperature drop across the model
    ndict['tempdrop'] = arg['rayleigh']*arg['refvisc']*arg['thermdiff'] \
        / (arg['density']*arg['thermexp']*arg['gravacc']*arg['radius']**3)

    # thermal conductivity
    ndict['thermcond'] = arg['thermdiff']*arg['density']*arg['cp']

    # scalings for non-dimensional to dimensional quantities
    # thermal diffusion timescale (Myr)
    ndict['myr2sec'] = 1e6*365.25*24*3600
    ndict['scalet'] = arg['radius']**2/(arg['thermdiff']*ndict['myr2sec'])

    # velocity scaling (used to be known as base_scale)
    # non-dimensional to cm/yr
    ndict['scalev'] = arg['thermdiff']/arg['radius']
    ndict['scalev'] *= 100*3600*24*365.25

    # surf_topography - uses density_above for seawater!
    # might need scaling by refstate.rho and refstate.gravity
    # for compressible
    ddensity = arg['density'] - arg['density_above']
    ndict['scalest'] = arg['refvisc']*arg['thermdiff']
    ndict['scalest'] /= (ddensity*arg['gravacc']*arg['radius']**2)

    # botm_topograpy - uses density_below
    # might need scaling by refstate.rho and refstate.gravity
    # for compressible
    ddensity = arg['density_below'] - arg['density']
    ndict['scalebt'] = arg['refvisc']*arg['thermdiff']
    ndict['scalebt'] /= (ddensity*arg['gravacc']*arg['radius']**2)

    # heat flux
    ndict['scalehf'] = ndict['thermcond']*ndict['tempdrop']/arg['radius']
    ndict['scalehf'] *= 1000 # Watts to milliwatts

    # pressure (also stress) scaling
    ndict['scalep'] = arg['refvisc']*arg['thermdiff']
    ndict['scalep'] /= (arg['radius']**2)

    # some of these values are only written to the CitcomS pid file
    # for a regional model (nproc_surf = 1).  However, let us determine
    # the domain for all cases including a global model.
    ndict['theta_min'] = arg.get('theta_min', 0.0)
    ndict['theta_max'] = arg.get('theta_max', np.pi)
    ndict['fi_min'] = arg.get('fi_min', 0.0)
    ndict['fi_max'] = arg.get('fi_max', 2.0*np.pi)
    ndict['lat_min'] = 90 - np.degrees( ndict['theta_max'] )
    ndict['lat_max'] = 90 - np.degrees( ndict['theta_min'] )
    ndict['lon_min'] = np.degrees( ndict['fi_min'] )
    ndict['lon_max'] = np.degrees( ndict['fi_max'] )

    # useful flags
    # N.B. 'FULL_SPHERE' because global is a python keyword
    if arg['nproc_surf'] == 1:
        ndict['REGIONAL'] = True
        ndict['FULL_SPHERE'] = False
    elif arg['nproc_surf'] == 12:
        ndict['REGIONAL'] = False
        ndict['FULL_SPHERE'] = True

    return ndict

#=====================================================================
#=====================================================================
#=====================================================================
def get_proc_list_from_cap_list( arg, cap_list ):

    '''Map a list of data by cap to a list of data by processor.  This
       is effectively an autoUNcombine(.py) operation.'''

    logging.info('start get_proc_list_from_cap_list') 

    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']
    nprocx = arg['nprocx']
    nprocy = arg['nprocy']
    nprocz = arg['nprocz']
    nx = arg['nx']
    ny = arg['ny']
    nz = arg['nz']
    proc_node = arg['proc_node']
    nproc_surf = arg['nproc_surf']

    proc_list = []

    for cc in range( nproc_surf ):
      cap_data = cap_list[cc]
      for jj in range( nprocy ):
        for ii in range( nprocx ):
          for kk in range( nprocz ):
            levy = jj * (ny-1) # y level
            levx = ii * (nx-1) # x level
            levz = kk * (nz-1) # Z level
            proc_data = []
            for yy in range( ny ):
              for xx in range( nx ):
                for zz in range( nz ):
                  cnode = zz+levz + (levx+xx)*nodez + (levy+yy)*nodex*nodez
                  proc_data.append( cap_data[cnode] )
            proc_list.append( proc_data )

    return proc_list

#=====================================================================
#=====================================================================
#=====================================================================
def read_cap_files_to_cap_list( pid_d, filename ):

    '''Read multiple cap files and return the data as a float
       tuple in a list by cap.'''

    print( Core_Util.now(), 'read_cap_files_to_cap_list:' )

    nproc_surf = pid_d['nproc_surf']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    nodez = pid_d['nodez']

    cap_names, header_list = define_cap_or_proc_names( pid_d, filename, 'cap' )

    cap_list = []

    for cc in range( nproc_surf ):
        cap_list.append([])
        cap_name = cap_names[cc]
        print( Core_Util.now(), 'reading cap:', cap_name)
        cap_file = open( cap_name, 'r' )

        # check for header (only one line)
        first_line = cap_file.readline().rstrip('\n')
        if not first_line in header_list:
            cols = first_line.split()
            line_list = tuple([float(val) for val in cols])
            cap_list[cc].append( line_list )

        # now loop over data
        for line in cap_file:
            cols = line.split()
            line_list = tuple([float(val) for val in cols])
            cap_list[cc].append( line_list )

        cap_file.close()

    return cap_list

#=====================================================================
#=====================================================================
#=====================================================================
def read_cap_z_coor( arg, filename ):

    '''Read CitcomS radius from cap file.'''

    if verbose: print( Core_Util.now(), 'read_cap_z_coor:' )

    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']

    coor = []
    cap_names, header_list = define_cap_or_proc_names( arg, filename, 'cap' )

    cap_file = open( filename, 'r' )
    for node, line in enumerate( cap_file ):
        if line.rstrip('\n') in header_list:
            if verbose: print( Core_Util.now(), 'strip header:', line)
            continue
        else:
            val = line.split()[2]
            coor.append( float(val) )
        if node == nodez: break

    return coor

#=====================================================================
#=====================================================================
#=====================================================================
def read_citcom_coor_type( arg, filename='' ):

    '''Automagically determine the CitcomS coordinate file type based
       on the number of lines in the file.  Will try to read from
       user-specified path first and then try [datadir]/[proc]/
       [datafile].coord.[proc] second.'''

    logging.debug('start reading citcom coor type' )

    datadir = arg['datadir']
    datafile = arg['datafile']
    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']
    nx = arg['nx']
    ny = arg['ny']
    nz = arg['nz']

    # plus integers are header lines
    options = {nodex + nodey + nodez + 3 : 'regional',
               nodez + 1                 : 'global',
               nodex * nodey * nodez + 1 : 'cap',
               nx * ny * nz + 1          : 'proc'     }

    # replace # with 0 to read first coordinate file
    for pp in range(4,0,-1):
        filename2 = filename.replace( '#'*pp, '0'*pp)

    # first, try user-specified template
    try:
        infile = open( filename2, 'r' )
    except FileNotFoundError:
        logging.warning(f'cannot find file {filename2}. try pid data directory again.' )
        # second, try pid data directory for *.coord.* files
        # of the form: [datadir]/[proc]/[datafile].coord.[proc]
        try:
            filename = datadir.replace( '%RANK', '#' ) + '/' + datafile
            filename += '.coord.#' # to return as template
            filename3 = datadir.replace( '%RANK', '0' ) + '/' + datafile
            filename3 += '.coord.0' # to check it exists
            infile = open( filename3, 'r' )
        except FileNotFoundError:
            logging.critical(f'cannot find either "{filename2}" or "{filename3}".' )
            logging.critical('at lease one of the files is required!')
            logging.info('check the "coord_dir" parameter in preprocessing input file and' + 
                    ' "datafile" parameter in citcoms input file!' )
            sys.exit( 1 )

    num_lines = sum(1 for line in infile)
    infile.close()

    logging.debug(f'read_citcom_coor_type: number of lines= { num_lines }' )
    
    try:
        mytype = options[num_lines]
    except KeyError:
        #print(traceback.format_exc())
        logging.critical(f'the line number {num_lines} must be one of these numbers ' + 
                f'{nodex + nodey + nodez + 3, nodez + 1 , nodex * nodey * nodez + 1, nx * ny * nz + 1}')
        sys.exit( ' - XXXX - Unknown coordinate file format!!' )

    # if the user-specified files did not exist, filename may have
    # been updated to a new template based on the datadir location.
    # Therefore return filename so subsequent functions know the
    # new template.
    return mytype, filename

#=====================================================================
#=====================================================================
#=====================================================================
def read_citcom_z_coor( arg, filename='' ):

    '''Read any type of citcoms coordinate file (regional, global,
       cap (0), *.coord.* from processor (0)), and compute radius,
       radius_km, depth, depth_km.  If processor, all processor
       *.coord.* files upto proc=nodez-1 must be at the same location
       (path).'''

    logging.info('start reading citcom z coor')

    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']
    nx = arg['nx']
    ny = arg['ny']
    nz = arg['nz']
    radius = arg['radius']
    radius_outer = arg['radius_outer']

    funct = {'regional' : read_regional_z_coor,
             'global'   : read_global_z_coor,
             'cap'      : read_cap_z_coor,
             'proc'     : read_proc_z_coor    }   

    rad = funct[arg['coord_type']]( arg, arg['coord_file_in_use'] )

    # non-dimensional radius (to 6 d.p.s)
    rad = np.around(np.array(rad),6).tolist()
    # dimensional radius in km, round then to nearest int
    rad_km = radius*1E-3*np.array(rad)
    rad_km = np.around(rad_km,0)
    rad_km = rad_km.tolist()
    # non-dimensional depth (to 6 d.p.s)
    dep = radius_outer-np.array(rad)
    dep = np.around(dep,6).tolist()
    # dimensional depth in km, round then to nearest int
    dep_km = radius*1E-3*np.array(dep)
    dep_km = np.around(dep_km,0)
    dep_km = dep_km.tolist()

    # update the master dictionary
    ndict = {'radius':rad,'radius_km':rad_km,
             'depth':dep,'depth_km':dep_km}

    return ndict

#=====================================================================
#=====================================================================
def get_znode_from_depth( z_d, request_depth ):
    ''' from a dictionary of z info from read_citcom_z_coor() return the closest znode for the requested depth'''

    # the node to return 
    znode = 0

    # NOTE search from deep to shallow, matching the order of the lists in z_d 
    depth_km = z_d['depth_km']
    # check for depths above the top 
    if request_depth > depth_km[0]:
        znode = 0
        print(now(), 'get_znode_from_depth: WARNING: request_depth > deepest level, using this value: depth = ', depth_km[0], '; znode =', znode)

    elif request_depth < depth_km[-1] :
        znode = len( z_d['depth_km'] )
        print(now(), 'get_znode_from_depth: WARNING: request_depth < shallowest level, using this value: depth = ', depth_km[-1], '; znode =', znode)
    
    else :
        # Use list comprehension to compute the difference between test and values in list
        # NOTE: the if clause > 0 , because depth_km runs from deep (larger numbers) to shallow (small numbers)
        delta_list = [ d - request_depth for d in depth_km if d - request_depth > 0 ]

        if verbose: 
            print( now(), 'get_znode_from_depth: len(delta_list) =', len(delta_list))
            print( now(), 'get_znode_from_depth: delta_list =', delta_list)
            print( now(), 'get_znode_from_depth: len(depth_km) =', len(depth_km))

        # get index of, and delta from value to test for previous value
        prev_i = len(delta_list) - 1
        prev_delta = delta_list[-1]

        # get index of, and delta from value to test for next value
        next_i = len(delta_list)
        next_delta = depth_km[next_i] - request_depth

        # index into main list of timesteps
        i = 0
        if abs(prev_delta) < abs(next_delta) :
            i = prev_i
        else: 
            i = next_i

        # set the znode
        znode = i

        if verbose:
            print(now(), 'get_znode_from_depth:', 'request_epth=', request_depth)
            print(now(), 'get_znode_from_depth:', 'prev=', depth_km[prev_i],'delta=', prev_delta, 'i=', prev_i )
            print(now(), 'get_znode_from_depth:', 'next=', depth_km[next_i],'delta=', next_delta,'i=', next_i )
            print(now(), 'get_znode_from_depth:', 'depth=', depth_km[i])
            print(now(), 'get_znode_from_depth:', 'znode=', znode)

    # end of if / elif / else  to set znode
    return znode

#=====================================================================
def read_citcom_surface_coor( arg, filename='' ):

    '''Read CitcomS surface (i.e., theta, phi) coordinates from either
       cap files or processor files and return the lon (0-360), lat in
       a cap list which contains the data for each cap.  Specify the
       filename for cap (0) or processor (0) in the argument.  All cap
       files upto nproc_surf, or all processor *.coord.* files upto
       total_proc, must be at the same location (path).'''

    logging.info('start read_citcom_surface_coor' )

    nproc_surf = arg['nproc_surf']
    radius = arg['radius']
    nodez = arg['nodez']

    funct = { 'cap'      : read_cap_files_to_cap_list,
              'proc'     : read_proc_files_to_cap_list }
    
    try:
        cap_list = funct[arg['coord_type']]( arg, arg['coord_file_in_use'], 'coord' )
    except KeyError:
        print('Filename must be for a cap (0) or processor (0)')
        sys.exit(1)

    surf_list = []

    for cc in range(nproc_surf):
        surf_list.append([])
        logging.debug(f'processing: cap{cc}')
        # skip over nodez for theta, phi only
        for cols in cap_list[cc][0::nodez]:
            theta = cols[0]
            phi = cols[1]
            lat = 90 - np.degrees(theta)
            lon = np.degrees(phi)
            surf_list[cc].append( (lon, lat) )

    return surf_list

#=====================================================================
#=====================================================================
#=====================================================================
def read_global_z_coor( arg, filename ):

    '''Read CitcomS radius from global coordinate file.'''

    if verbose: print( Core_Util.now(), 'read_global_z_coor:' )

    coor = []
    header = 'nodez='

    infile = open(filename,'r')
    for line in infile:
        if line.startswith( header ):
            if verbose: print( Core_Util.now(), 'strip header:',line)
            continue
        else:
            node, val = line.split()
            coor.append( float(val) )

    return coor

#=====================================================================
#=====================================================================
#=====================================================================
def read_proc_files_to_cap_list( arg, filename, field_name ):

    '''Read lines of data (e.g., coord, velo) from many CitcomS
       processor files and return the data in a cap list which
       contains the data lists for each cap.  All proc files must be 
       at the same location (path).'''

    # FIXME: remove
    #global verbose
    #verbose = True
    #if 'coord' in filename:
    #    verbose = False

    logging.info('start read_proc_files_to_cap_list' )

    nproc_surf = arg['nproc_surf']
    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']
    nprocx = arg['nprocx']
    nprocy = arg['nprocy']
    nprocz = arg['nprocz']
    nx = arg['nx']
    ny = arg['ny']
    nz = arg['nz']
    total_proc = arg['total_proc']
    total_node = arg['total_node']


    # Get the proc names and list of potential headers for this filename type 
    proc_names, header_list = define_cap_or_proc_names( arg, filename, 'proc' )

    logging.debug(f'read_proc_files_to_cap_list: filename = {filename}')
    logging.debug(f'read_proc_files_to_cap_list: field_name = {field_name}')
    logging.debug(f'read_proc_files_to_cap_list: proc_names = {proc_names}')
    logging.debug(f'read_proc_files_to_cap_list: header_list = {header_list}')

    # Set up an empty cap list to populate 
    cap_list = []

    # Loop over 0 .. nproc_surf
    for nn in range( nproc_surf ):

      logging.debug( '======================================')
      logging.debug( f'fread_proc_files_to_cap_list: nn = {nn}')

      # Set up lists to store data for each cap
      cap_data = [0 for cc in range(nodex*nodey*nodez)]
      logging.debug('read_proc_files_to_cap_list: len(cap_data) = {len(cap_data)}')

      for jj in range( nprocy ):
        for ii in range( nprocx ):
          for kk in range( nprocz ):

            # global processor number
            gp = kk + (ii*nprocz) + (jj*nprocz*nprocx) + (nn*nprocz*nprocx*nprocy)
            levy = jj*(ny-1)
            levx = ii*(nx-1)
            levz = kk*(nz-1)

            gpname = proc_names[gp]


            # flag to determine if this is surf or botm data
            botm_or_surf_flag = False 
            if 'surf' in gpname: 
                botm_or_surf_flag = True
            if 'botm' in gpname:
                botm_or_surf_flag = True

            # file might not exist for this processor if *.surf.* or *.botm.* data
            try:
                # open the file and read all lines into a list
                gpfile = open( gpname, 'r' )
                lines = gpfile.readlines()

                # Get the header count for this field
                header_count = field_to_file_map[field_name]['header_count']

                # remove header lines 
                lines = lines[ header_count: ]

                # close up shop
                gpfile.close()
                logging.debug('read_proc_files_to_cap_list: reading processor data from file : {gpname}')

            # Usually this clause will be called when reading CitcomS surface data
            except FileNotFoundError:
                # *.surf.* and *.botm* do not exist for all processors, pad with zeroes
                lines = tuple(['0.0 0.0 0.0 0.0\n' for x in range( nx*ny )]) # must be string for split() to work
                print( Core_Util.now(), 'read_proc_files_to_cap_list: padding data ; missing file file: ', gpname)


            #if verbose: print( Core_Util.now(), 'read_proc_files_to_cap_list: read in data volume: len(lines) =', len(lines))

            # SURFACE DATA

            # The *.surf.* and *.botm.* files will only have nx*ny entries because they only contain
            # data for one radial plane.  let's map the data to volume regardless by determining
            # the global z node for either the top (nodez-1) or bottom (0)
            if botm_or_surf_flag :
                # process surface data into volume data
                for yy in range( ny ):
                  for xx in range( nx ):

                    # NOTE: surface data, only xx and yy are relevant for pnn 
                    # processor node number
                    pnn = xx + yy*nx

                    # extract the data from the file data 
                    cols = lines[pnn].split()

                    # create empty_cols to create full volume data set
                    empty_cols = tuple( 0.0 for i in range(len(cols)) )

                    # loop over z 
                    for zz in range ( nz ): 

                        # cap node number
                        gny = yy + levy
                        gnx = xx + levx
                        gnz = zz + levz

                        # save for refernece:
                        #if 'botm' in gpname: #    gnz = 0
                        #if 'surf' in gpname: #    gnz = nodez-1

                        # compute global node number 
                        gnn = gnz + gnx*nodez + gny*nodez*nodex

                        # check for botm slices 
                        if gnz == 0: 
                          # re-assign proc data from file to cap data using the global node number
                          cap_data[gnn] = tuple([float(val) for val in cols])

                        # check for surf slices 
                        elif gnz == nodez-1: 
                            # re-assign proc data from file to cap data using the global node number
                            cap_data[gnn] = tuple([float(val) for val in cols])

                        else: 
                            # assign empty proc data to cap cap using the global node number
                            cap_data[gnn] = tuple([float(val) for val in empty_cols])

                        # FIXME: remove once testing is complete 
                        #if verbose: print( Core_Util.now(), 'read_proc_files_to_cap_list: SURF: pnn = ', pnn, '; gnn=', gnn, '; cols = ', cols, '; len(cap_data[gnn]) = ', len(cap_data[gnn]) )

            # VOLUME DATA
            else:
                # loop over lines
                for yy in range( ny ):
                  for xx in range( nx ):
                    for zz in range( nz ):
                      # processor node number
                      pnn = zz + xx*nz + yy*nz*nx
                      # cap node number
                      gny = yy + levy
                      gnx = xx + levx
                      gnz = zz + levz
                      gnn = gnz + gnx*nodez + gny*nodez*nodex

                      # extract the data from the file data 
                      cols = lines[pnn].split()
                      # re-assign proc data from file to cap data using the global node number
                      # XXX DJB - FYI MARK
                      # HERE WE CAN EXTRACT THE COLUMN FROM THE DATA AND STORE AS CAP_DATA IF WE WANT
                      # BUT THIS WILL MEAN WE MIGHT HAVE TO LOOP OVER THE SAME DATA TO GET VX, VY, VZ ETC
                      # TO DISCUSS WITH MARK
                      cap_data[gnn] = tuple([float(val) for val in cols])

                      # FIXME: remove once testing is complete 
                      #if verbose: print( Core_Util.now(), 'read_proc_files_to_cap_list: VOLUME: pnn = ', pnn, '; gnn=', gnn, '; cols = ', cols, '; len(cap_data[gnn]) = ', len(cap_data[gnn]) )

            # end of check on SURF vs VOLUME data 

          # End of Loop: for kk in range( nprocz ):
        # End of Loop: for ii in range( nprocx ):
      # End of Loop: for jj in range( nprocy ):
      
      logging.debug(' ')
      logging.debug('read_proc_files_to_cap_list: CAP summary: ')
      logging.debug(f'read_proc_files_to_cap_list: len(cap_data) = {len(cap_data)}' )

      cap_list.append( cap_data )

    # End of loop: for nn in range( nproc_surf ):
    logging.info(f'read_proc_files_to_cap_list: CAP LIST summary (End of loop: for nn in range( nproc_surf ): ')
    logging.info(f'read_proc_files_to_cap_list: len(cap_list) = {len(cap_list)}' )
    return cap_list

#=====================================================================
#=====================================================================
#=====================================================================
def read_proc_z_coor( arg, filename ):

    '''Read CitcomS radius from several processor *.coord.* files.'''

    logging.info('start reading proc z coor' )

    nprocz = arg['nprocz']
    nodez = arg['nodez']
    nz = arg['nz']

    prefix = filename[:-1]
    coor = []

    proc_names, header_list = define_cap_or_proc_names( arg, filename, 'proc' )

    for procz in range(nprocz):
        coorname = proc_names[procz]
        logging.info(f'read_proc_z_coor: reading {coorname}')
        levz = (nz-1)*procz
        infile = open(coorname,'r')
        for lnz, line in enumerate(infile):
            if line.rstrip('\n') in header_list:
                logging.debug("strip header: 'line.rstrip('\n')")
                continue
            else:
                val = line.split()[2]
                gnz = lnz + levz
                coor.append( float(val) )
            if lnz == nz-1 and procz != nprocz-1: break
            if gnz == nodez and procz == nprocz-1: break

    return coor

#=====================================================================
#=====================================================================
#=====================================================================
def read_regional_z_coor( arg, filename ):

    '''Read CitcomS radius from regional coordinate file.'''

    if verbose: print( Core_Util.now(), 'read_regional_z_coor:' )

    nodex = arg['nodex']
    nodey = arg['nodey']
    nodez = arg['nodez']

    coor = []
    infile = open(filename,'r')
    for num, line in enumerate(infile):
        if num < nodex + nodey + 3:
            continue
        else:
            node, val = line.split()
            coor.append( float(val) )

    return coor

#=====================================================================
#=====================================================================
#=====================================================================
def write_cap_or_proc_list_to_files( arg, filename, inp_list, \
        cap_or_proc, head_flag ):

    '''Write cap or proc lists to files.  inp_list is an iterable
       (e.g., list or tuple) of the data lists by cap that will be 
       merged during the output.  Use a single tuple i.e.,
       inp_list = (X,) to pass only one list to this function, where
       X is the list for output.'''

    logging.info('start write_cap_or_proc_list_to_files' )

    out_names, header_list = define_cap_or_proc_names( arg, filename, cap_or_proc )

    # zip the cap data together first
    zipcap = list(zip(*inp_list))

    head = ''
    # write out standard CitcomS header
    if head_flag:
        if cap_or_proc == 'cap':
            nodex = arg['nodex']
            nodey = arg['nodey']
            nodez = arg['nodez']
            if len(inp_list[0][0]) == nodex*nodey*nodez:
                # all nodes
                head = header_list[0]+'\n'
            elif len(inp_list[0][0]) == nodex*nodey:
                # surface nodes only
                head = header_list[1]+'\n'
        # XXX coord header is only one line (this always gives two)
        elif cap_or_proc == 'proc':
            head = '\n'.join( header_list ) + '\n'

    # write list data
    for nn in range( len(inp_list[0]) ):

        out_name = out_names[nn]
        logging.debug(f'writing: {out_name}' )
        out_file = open( out_name, 'w' )
        out_file.write( head )

        # zip the list entries for this cap
        ziplines = list(zip(*zipcap[nn]))

        for line_tuple in ziplines:
            # need to flatten entire line bc e.g. coordinate data
            # is stored as a 2-tuple but some lists may just contain
            # single values (floats or ints)
            line_tuple = Core_Util.flatten( line_tuple )
            out_line = ' '.join( map( str, line_tuple ) ) + '\n'
            out_file.write( out_line )
        out_file.close()

    return out_names

#=====================================================================
#=====================================================================
#=====================================================================
def get_time_spec_dictionary(time_spec, time_d = False) :
    '''Process a time_spec string and return a dictionary, spec_d, with this data:

    spec_d['time_spec'] = the original time_spec as a string
    spec_d['time_list'] = the list of specific times from processing the spec, as string with or without units

  If the optional 'time_d' argument is given with a valid Citcoms time dictionary, 
  then this function will generate these lists:

    spec_d['age_Ma'] = the list of closest times in reconstruction ages (Ma)
    spec_d['runtime_Myr'] = the list of closest times in model run time (Myr)
    spec_d['timestep'] = the list of closest times in model time steps (non-dimensional).

A time_spec string may be given as: 

    a single item, without units for model time steps, with Ma or Myr for dimensional time
        e.g.: '1000', or '250Ma', or '100 Myr'

    a comma delimited list of times, without units for model time steps, with Ma or Myr for dimensional time
        comma lists may have values of multiple units 
        e.g.: '1000, 170Ma,100 Myr,1600'

    a slash delimited start/end/step set, without units for model time steps, with Ma or Myr for dimensional time
        e.g.: '250/25/25Ma'

    or a filename with indidual single value entries, with or without units, one item per line.

    NOTE: spec_d['time_list'] is a list of strings; client code will have to strip units and convert to numbers.'''

    # convert the input to a string
    time_spec = str(time_spec)

    if verbose: print( Core_Util.now(), 'get_time_spec_dictionary: time_spec =', time_spec )

    # create an empty spec dictionary to populate and return
    spec_d = { 'time_spec': time_spec, 'time_list': [] } 

    # the list of times from parsing a time_spec string
    time_list = [] 

    # check for discrete times separated via comma 
    if time_spec.count(','):
        time_spec = time_spec.replace('[', '')
        time_spec = time_spec.replace(']', '')
        time_list = time_spec.split(',')
        time_list = [ i.strip() for i in time_list ] 
        time_list = [ i.rstrip() for i in time_list ]
        if verbose: print( Core_Util.now(), 'get_time_spec_dictionary: time_list =', time_list )

    # check for sequence of times separated via / 
    elif time_spec.count('/'): 
        if len( time_spec.split('/') ) != 3 :
            msg = "Three numbers are required: start/end/step"
            raise ValueError(msg)

        (start,end,step) = time_spec.split('/') 

        # strip units, make floating point numbers for looping math
        if start.endswith('Ma') or end.endswith('Ma') or step.endswith('Ma'):
            units = 'Ma'
            start = int(start.replace('Ma', ''))
            end = int(end.replace('Ma', ''))
            step = int(step.replace('Ma', ''))
        elif start.endswith('Myr') or end.endswith('Myr') or step.endswith('Myr'):
            units = 'Myr'
            start = int(start.replace('Myr', ''))
            end = int(end.replace('Myr', ''))
            step = int(step.replace('Myr', ''))
        else:
            units = ''
            start = int(start)
            end = int(end)
            step = int(step)

        # build time_list, add back units
        if (start < end):
            t = start
            while t <= end:
                time_list.append(str(t) + units)
                t = t + step
        elif (start > end):
            t = start
            while t >= end:
                time_list.append(str(t) + units)
                t = t - step

    # check for data file 
    elif time_spec.endswith('.dat'):
        time_list = Core_Util.parse_general_one_value_per_line_type_file(time_spec)

    # else, single time spec
    else :
        time_list = [ time_spec ]

    if verbose: print( Core_Util.now(), 'get_time_spec_dictionary:', time_list )

    # update time_list
    spec_d['time_list'] = time_list

    # If a valid citcoms time dictionary is passed in, then generate the additional lists
    if time_d: 

      # empty lists to fill
      timestep_l = []
      age_l = []
      runtime_l = []

      # process each time on the time_list to get the three equivalent units
      for t in spec_d['time_list']:

        # check the units on t:
        if t.endswith('Ma'):

            # TODO: grid_maker.py should not require a citcom time file when we are generating
            # vx, vy, vmag for gplates data only.  This try statement is a quick work around but
            # should be cleaned up
            try:
                trip = get_time_triple_from_age( time_d['triples'] , float(t.replace('Ma','') ) )
            except TypeError:
                return spec_d

            timestep_l.append( trip[0] )
            age_l.append(      trip[1] )
            runtime_l.append(  trip[2] )

        elif t.endswith('Myr'):

            trip = get_time_triple_from_runtime( time_d['triples'] , float(t.replace('Myr','') ) )

            timestep_l.append( trip[0] )
            age_l.append(      trip[1] )
            runtime_l.append(  trip[2] )

        else:

            trip = get_time_triple_from_timestep( time_d['triples'] , float(t) )

            timestep_l.append( trip[0] )
            age_l.append(      trip[1] )
            runtime_l.append(  trip[2] )

      # update the data 
      spec_d['age_Ma'] = age_l
      spec_d['runtime_Myr'] = runtime_l
      spec_d['timestep'] = timestep_l

    # end of check for time_d

    return spec_d

#=====================================================================
#=====================================================================
#=====================================================================
def read_citcom_time_file(pid_d):
    '''read a citcoms .time file and generate a dictionary of lists of time values:
    time_d['step'] = [list of model time steps]
    time_d['age_Ma'] = [list of reconstruction ages in Ma]
    time_d['runtime_Myr'] = [list of runtime value in Myr]
    time_d['triples'] = [list of time triple values (model steps, age in Ma, runtime in Myr) ]
''' 
    if verbose: print( Core_Util.now(), 'read_citcom_time_file:' ) 

    # get basic info about the model run
    datadir       = pid_d['datadir']
    datafile      = pid_d['datafile']
    start_age     = pid_d['start_age']
    output_format = pid_d['output_format']

    # compute temperature scale factor
    layer_km = pid_d['radius_km']
    scalet   = pid_d['scalet']

    # check for time file base name in local directory
    timefile_name = '%s.time' % datafile 
    
    # check local dir first
    print(now(), 'Check for time file:', timefile_name)
    if not os.path.exists(timefile_name):
        print( now(), "NOT found" )
        # prepend sub dirs if local file not found:
        timefile_name = datadir.replace( '%RANK', '0' ) + '/' + timefile_name

    # open the file 
    print(now(), 'Check for time file:', timefile_name)
    time_file = None
    try :
        time_file = open(timefile_name)
    except:
        print(now(), 'WARNING: Time file not found:', timefile_name)
        return None

    # empty lists
    age_l = []
    step_l = []
    runtime_l = []
    triple_l = []

    # read the file
    try: 
        lines = time_file.read().splitlines()
    finally:
        time_file.close()
 
    therm_diff = pid_d['thermdiff']
    scale_t = pid_d['scalet']

    # loop over time file
    for line in lines:

        # parse the data line and append the step
        step, t, dt, CPUtime, CPUdt = line.split()
        step_l.append( int(step) )

        # compute age in Ma and append it
        age = start_age - ( float(t) * scale_t)
        age_l.append( float(age) )
  
        # compute runtime in Myr and append it
        runtime = float(t) * scale_t
        runtime_l.append( float(runtime) )

        # compose the triple and append it
        triple_l.append( ( int(step), float(age), float(runtime) ) )

    # create the dictionary to return 
    d = { 'age_Ma' : age_l, 
          'step' : step_l,
          'runtime_Myr' : runtime_l, 
          'triples' : triple_l }

    if verbose: print( Core_Util.now(), 'read_citcom_time_file: d = ' ) 
    if verbose: print( Core_Util.tree_print(d) )

    return d

#=====================================================================
#=====================================================================
#=====================================================================
def get_time_triple_from_age(triple_list, test_age):
    '''locate and return the closest time triple for given age in Ma
from the list of tuples: (step, age, runtime) crated from a citcom time file.'''

    # check bounds 
    max_age = triple_list[0][1]
    min_age = triple_list[-1][1]

    if (test_age > max_age):
        msg = 'age %f Ma > max age %f Ma' % (test_age, max_age)
        raise IndexError(msg)

    if (test_age < min_age):
        # FIXME: do we always want to print a NOTE that the requested test_age is being reset?
        print('NOTE: test_age %f Ma < min age %f Ma; resetting test_age = min age' % (test_age, min_age))
        test_age = min_age

    # shortcut for exact values
    for (s,a,r) in triple_list:
        if a == test_age:
            if verbose: 
                print( Core_Util.now(),'get_time_triple_from_age: (s,a,r) =',(s,a,r) )
            return (s,a,r)

    # list comprehension to compute the difference between test_age and model ages
    # NOTE the if a-test_age > 0
    delta_list = [a-test_age for (s,a,r) in triple_list if a-test_age > 0 ]

    # get index of, and delta from value to test for previous value
    prev_i = len(delta_list) - 1
    prev_delta = delta_list[-1]

    # get index of, and delta from value to test for next value
    next_i = len(delta_list)
    next_delta = triple_list[next_i][1] - test_age

    # index smallest delta and index into list of tuples
    if abs(prev_delta) < abs(next_delta) :
        i = prev_i
    else: 
        i = next_i

    # FIXME: ?
    i = prev_i

    if verbose:
        print( Core_Util.now(), 'get_time_triple_from_age:', 'prev=', triple_list[prev_i], 'delta=', prev_delta, 'i=', prev_i )
        print( Core_Util.now(), 'get_time_triple_from_age:', 'test=', test_age )
        print( Core_Util.now(), 'get_time_triple_from_age:', 'trip=', triple_list[i] )
        print( Core_Util.now(), 'get_time_triple_from_age:', 'next=', triple_list[next_i], 'delta=', next_delta, 'i=', prev_i )

    return triple_list[i]

#=====================================================================
#=====================================================================
#=====================================================================
def get_time_triple_from_runtime(triple_list, test_runtime):
    '''locate and return the closest time triple for the given runtime in Myr,
 from the list of tuples: (step, age, runtime) crated from a citcom time file.'''

    # check bounds : NOTE indices
    max_runtime = triple_list[-1][2]
    min_runtime = triple_list[0][2]
    if (test_runtime > max_runtime):
        msg = 'test_runtime %f Myr > max runtime %f Myr' % (test_runtime, max_runtime)
        raise IndexError(msg)
    if (test_runtime < min_runtime):
        msg = 'test_runtime %f Myr < min runtime %f Myr' % (test_runtime, min_runtime)
        raise IndexError(msg)

    # shortcut for exact values
    for (s,a,r) in triple_list:
        if float(r) == float(test_runtime):
            if verbose: print( Core_Util.now(),'get_time_triple_from_runtime: (s,a,r) =', (s,a,r) )
            return (s,a,r)

    # list comprehension to compute the difference between test and values in list
    # NOTE : the if clause < 0
    delta_list = [r-test_runtime for (s,a,r) in triple_list if r-test_runtime < 0 ]

    # get index of, and delta from value to test for previous value
    prev_i = len(delta_list) - 1
    prev_delta = delta_list[-1]

    # get index of, and delta from value to test for next value
    next_i = len(delta_list)
    next_delta = triple_list[next_i][1] - test_runtime

    # index into main triple_list of tuples
    i = 0
    if abs(prev_delta) < abs(next_delta) :
        i = prev_i
    else: 
        i = next_i

    if verbose:
        print( Core_Util.now(), 'get_time_triple_from_runtime:', 'prev=', triple_list[prev_i],'delta=', prev_delta, 'i=', prev_i )
        print( Core_Util.now(), 'get_time_triple_from_runtime:', 'test=', test_runtime )
        print( Core_Util.now(), 'get_time_triple_from_runtime:', 'time=', triple_list[i] )
        print( Core_Util.now(), 'get_time_triple_from_runtime:', 'next=', triple_list[next_i],'delta=', next_delta,'i=', prev_i )

    return triple_list[i]

#=====================================================================
#=====================================================================
#=====================================================================
def get_time_triple_from_timestep(triple_list, test_step):
    '''locate and return the time triple for a given model time step '''

    # check bounds : NOTE indices
    min_step = triple_list[0][0]
    max_step = triple_list[-1][0]

    if (test_step > max_step):
        print( now(), 'get_time_triple_from_timestep: WARNING: test_step >= largest timestep (', max_step, '). Resetting to max value.')
        test_step = max_step
        #msg = 'test_step %i > max step %i' % (test_step, max_step)
        #raise IndexError(msg)
    if (test_step < min_step):
        print( now(), 'get_time_triple_from_timestep: WARNING: test_step <= smallest timestep (', min_step, '). Resetting to min value.')
        #msg = 'test_step %i < min step %i' % (test_step, min_step)
        #raise IndexError(msg)
        test_step = min_step

    # shortcut for exact values
    for (s,a,r) in triple_list:
        if int(s) == int(test_step):
            if verbose: print( Core_Util.now(),'get_time_triple_from_timestep: (s,a,r) =', (s,a,r) )
            return (s,a,r)

#=====================================================================
#=====================================================================
def find_available_timestep_from_timestep(master_d, field_name, request_timestep) :
    ''' From the specific citcom run data contained in master_d, and field,
locate the closest available time step data files avaialable for the requested timestep'''

    # the dictionary to return
    ret_d = {}

    # set up some local vars for this case 
    datadir  = master_d['pid_d']['datadir']
    datafile = master_d['pid_d']['datafile']

    # get the data file name specifics for this field 
    file_name_component = field_to_file_map[field_name]['file']

    # file name pattern to check ; NOTE: only checking proc 0 
    # check based on datafile and data dir like the timefile 
    file_patt = ''

    if os.path.exists( datadir + '/0/') :
        file_patt = datadir + '/0/' + datafile + '.' + file_name_component + '.0.*'

    elif os.path.exists( datadir + '/' ) :
        file_patt = datadir + '/' + datafile + '.' + file_name_component + '.0.*'

    elif os.path.exists('data') :
        file_patt = './data/0/' + datafile + '.' + file_name_component + '.0.*'

    elif os.path.exists('Data') :
        file_patt = './Data/0/' + datafile + '.' + file_name_component + '.0.*'

    # get a list of all the files that match the pattern
    # sorted by the time step values 
    file_list = glob.glob(file_patt) 
    step_list = [ int( f.split('.')[-1] ) for f in file_list ] 
    step_list = sorted( step_list )

    if verbose: 
        print( now(), 'find_available_timestep_from_timestep: file_patt =', file_patt)
        #print( now(), 'find_available_timestep_from_timestep: file_list =', file_list)
        print( now(), 'find_available_timestep_from_timestep: step_list =', step_list)

    # Get closest the time tripple for request_timestep
    time_triple = get_time_triple_from_timestep( master_d['time_d']['triples'], request_timestep )
    request_age = int( time_triple[1] )
    request_runtime =  time_triple[2]

    if verbose: 
        print( now(), 'find_available_timestep_from_timestep: request: time_triple =', time_triple)
        #print( now(), 'find_available_timestep_from_timestep: request_timestep =', request_timestep)
        #print( now(), 'find_available_timestep_from_timestep: request_runtime =', request_runtime)

    # Find closeset avalable time step to the request timestep 

    # reset the request_timestep for values < the smallest available
    if request_timestep <= step_list[0]:
        found_timestep = step_list[0]
        print()
        print( now(), 'find_available_timestep_from_timestep: WARNING: request_timestep <= smallest timestep:', step_list[0])
        print( now(), 'find_available_timestep_from_timestep: WARNING: Resetting request to that value.')
        print()

    # reset the request timestep for values > than the largest available
    elif request_timestep >= step_list[-1]:
        found_timestep = step_list[-1]
        print()
        print( now(), 'find_available_timestep_from_timestep: WARNING: request_timestep >= largest timestep:', step_list[-1])
        print( now(), 'find_available_timestep_from_timestep: WARNING: Resetting request to that value.')
        print()

    # somewhere in the middle - use lists to find where
    else :
        # Use list comprehension to compute the difference between test and values in list
        # NOTE: the if clause < 0 , because timestep runs from 0 on up ... 
        delta_list = [ s - request_timestep for s in step_list if s - request_timestep < 0 ]

        #if verbose: 
        #    print( now(), 'find_available_timestep_from_timestep: len(delta_list) =', len(delta_list))
        #    print( now(), 'find_available_timestep_from_timestep: delta_list =', delta_list)
        #    print( now(), 'find_available_timestep_from_timestep: len(step_list) =', len(step_list))

        # get index of, and delta from value to test for previous value
        prev_i = len(delta_list) - 1
        prev_delta = delta_list[-1]

        # get index of, and delta from value to test for next value
        next_i = len(delta_list)
        next_delta = step_list[next_i] - request_timestep

        # index into main list of timesteps
        i = 0
        if abs(prev_delta) < abs(next_delta) :
            i = prev_i
        else: 
            i = next_i

        # set the found timestep
        found_timestep = step_list[i]

        #if verbose:
        #    print( Core_Util.now(), 'find_available_timestep_from_timestep:', 'prev=', step_list[prev_i],'delta=', prev_delta, 'i=', prev_i )
        #    print( Core_Util.now(), 'find_available_timestep_from_timestep:', 'test=', request_timestep )
        #    print( Core_Util.now(), 'find_available_timestep_from_timestep:', 'time=', step_list[i] )
        #    print( Core_Util.now(), 'find_available_timestep_from_timestep:', 'next=', step_list[next_i],'delta=', next_delta,'i=', prev_i )

    # end of if / elif / else  to set found_timestep

    # get the age for this closest available time step
    time_triple = get_time_triple_from_timestep( master_d['time_d']['triples'], found_timestep )
    found_age = time_triple[1]
    found_runtime = time_triple[2]

    if verbose:
        print( now(), 'find_available_timestep_from_timestep:')
        print( now(), 'REQUEST: ', 
         'request_timestep = ', request_timestep, 
         '; request_age = ', request_age,      
         '; request_runtime = ', request_runtime )
        print( now(), 'FOUND:   ', 
         'found_timestep   = ', found_timestep,
         '; found_age   =', found_age,
         '; found_runtime   =', found_runtime)

    # create and return a dictionary with all the times 
    ret_d['request_timestep'] = request_timestep
    ret_d['request_age'] = request_age
    ret_d['request_runtime'] = request_runtime
    ret_d['found_timestep'] = found_timestep
    ret_d['found_age'] = found_age
    ret_d['found_runtime'] = found_runtime
 
    return ret_d

#=====================================================================
#=====================================================================
def find_available_timestep_from_age(master_d, field_name, request_age) :
    ''' From the specific citcom run data contained in master_d, and field,
locate the closest available time step data files avaialable for the requested age'''

    # get closest time step for requested age
    time_triple = get_time_triple_from_age( master_d['time_d']['triples'], request_age )
    request_timestep = int( time_triple[0] )
    request_runtime =  time_triple[2]

    # now call the main find function
    ret_d = find_available_timestep_from_timestep(master_d, field_name, request_timestep)
    return ret_d

#=====================================================================
#=====================================================================
def get_all_pid_data( pid_file ):
    '''This function generates a set of nested dictionary data 
holding the Citcom model run data (the pid data), the model time file, 
the model coordinate data, and, the geoframework default data.

The argument is a Citcom pid file.

This function will return the data in a top level master dictionary, 
with a set of nested dictionaries, with this structure:

    master['pid_d']      = dictionary of pid data,
    master['time_d']     = dictionary of time data,
    master['coor_d']     = dictionary of coordinate data,
    master['geoframe_d'] = dictionary of geoframework default data,
'''
    if verbose: print(Core_Util.now(), 'Core_Citcom.get_all_pid_data():')

    master_d = {}

    # Step 1: initialize the geodynamic framework config
    if verbose: print(Core_Util.now(), 'get_all_pid_data: Step 1: Read the geodynamic framework defaults')
    master_d['geoframe_d'] = Core_Util.parse_geodynamic_framework_defaults()

    # Step 2: read the CitcomS pid file 
    if verbose: print(Core_Util.now(), 'get_all_pid_data: Step 2: Read the CitcomS pid file')
    pid_d = Core_Util.parse_configuration_file( pid_file )

    if verbose: print(Core_Util.now(), 'get_all_pid_data: pid_d =')
    if verbose: Core_Util.tree_print(pid_d)

    extra_citcom_dict = derive_extra_citcom_parameters( pid_d )
    pid_d.update( extra_citcom_dict )
    master_d['pid_d'] = pid_d

    # Step 3: read the time file
    if verbose: print(Core_Util.now(), 'get_all_pid_data: Step 3: Read the time file')
    # this will be None if time file does not exist or cannot be found
    # NOTE: it is up to the client code to handle a missing time file.
    master_d['time_d'] = read_citcom_time_file( pid_d )

    # Step 4: read the coor data
    if verbose: print(Core_Util.now(), 'get_all_pid_data: Step 4: Read the coor data')
    if master_d['pid_d']['coor'] == 1:
        pid_d = master_d['pid_d']
        coor_file = master_d['pid_d']['coor_file']
        pid_d['coord_type'], pid_d['coord_file_in_use'] = read_citcom_coor_type( pid_d )
        coor_d = read_citcom_z_coor(pid_d, coor_file)
        master_d['coor_d'] = coor_d
 
    # return all data 
    return master_d
#=====================================================================
#=====================================================================
def dimensionalize_grid(pid_file, field_name, in_grid, out_grid) :
    '''Using the field_name and the coeficient and constants set in field_to_dimensional_map, dimensionalize in_grid and create out_grid'''

    if field_name not in field_to_dimensional_map.keys() :
        print(Core_Util.now(), 'ERROR: field_name', field_name, 'not found in field_to_dimensional_map()')
        return 

    # populate the map
    map = populate_field_to_dimensional_map_from_pid( pid_file )

    # get specific data for this field_name
    coef = map[field_name]['coef']
    const= map[field_name]['const']

    # now do some grid math with gmt 
    args = '%(in_grid)s %(coef)f MUL %(const)s ADD' % vars()
    Core_GMT.callgmt( 'grdmath', args, '', '=', out_grid )
#=====================================================================
#=====================================================================
def populate_field_to_dimensional_map_from_pid( pid_file ): 
    '''Set values in the global variable field_to_dimensional_map with specific coef and const values from data in the pid file'''

    global field_to_dimensional_map

    # get all the pid info 
    master_d = get_all_pid_data( pid_file )
    pid_d = master_d['pid_d']

    # extract values from master_d used to set specific coef and const for variables

    # FIXME place holder example XXX

    # temp 
    field_to_dimensional_map['temp']['coef'] = pid_d['tempdrop']
    # N.B. surfaceT is not used `dynamically' for Boussinesq models
    field_to_dimensional_map['temp']['const'] = pid_d['surfaceT']*pid_d['tempdrop']

    # velocity
    field_to_dimensional_map['vx']['coef'] = pid_d['scalev']
    field_to_dimensional_map['vx']['const'] = 0
    field_to_dimensional_map['vy']['coef'] = pid_d['scalev']
    field_to_dimensional_map['vy']['const'] = 0
    field_to_dimensional_map['vz']['coef'] = pid_d['scalev']
    field_to_dimensional_map['vz']['const'] = 0

    # pressure
    field_to_dimensional_map['pressure']['coef'] = pid_d['scalep']
    field_to_dimensional_map['pressure']['const'] = 0
 
    # surface 
    field_to_dimensional_map['surf_topography']['coef'] = pid_d['scalest']
    field_to_dimensional_map['surf_topography']['const'] = 0 # Nico to check
    field_to_dimensional_map['surf_heat_flux']['coef'] = pid_d['scalehf']
    field_to_dimensional_map['surf_heat_flux']['const'] = 0
    field_to_dimensional_map['surf_vel_colat']['coef'] = pid_d['scalev']
    field_to_dimensional_map['surf_vel_colat']['const'] = 0
    field_to_dimensional_map['surf_vel_lon']['coef'] = pid_d['scalev']
    field_to_dimensional_map['surf_vel_lon']['const'] = 0 
 
    # bottom
    field_to_dimensional_map['botm_topography']['coef'] = pid_d['scalebt']
    field_to_dimensional_map['botm_topography']['const'] = 0 # Nico to check
    field_to_dimensional_map['botm_heat_flux']['coef'] = pid_d['scalehf']
    field_to_dimensional_map['botm_heat_flux']['const'] = 0
    field_to_dimensional_map['botm_vel_colat']['coef'] = pid_d['scalev']
    field_to_dimensional_map['botm_vel_colat']['const'] = 0
    field_to_dimensional_map['botm_vel_lon']['coef'] = pid_d['scalev']
    field_to_dimensional_map['botm_vel_lon']['const'] = 0 

    # show the dict
    if verbose: print(Core_Util.now(), 'populate_field_to_dimensional_map_from_pid(): field_to_dimensional_map =')
    Core_Util.tree_print(field_to_dimensional_map)

    # return a copy of the adjusted map
    return dict( field_to_dimensional_map )

#=====================================================================
#=====================================================================
#=====================================================================
#=====================================================================
def report_on_model_run( cfg_file ):
    print( now(), 'report_on_model_rureport_on_model_run.py')

    # get the .cfg file as a dictionary
    control_d = Core_Util.parse_configuration_file( cfg_file )

    # set the pid file 
    pid_file = control_d['pid_file']

    # get the master dictionary and define aliases
    master_d = get_all_pid_data( pid_file )
    coor_d = master_d['coor_d']
    pid_d = master_d['pid_d']

    # set up working variables
    datafile = pid_d['datafile']
    depth_list = coor_d['depth_km']
    nodez = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']

    # Check how to read and parse the time spec:
    read_time_d = True
    for s in control_d['_SECTIONS_'] :
        if control_d[s]['field'].startswith('gplates_') :
            read_time_d = False
        
    # Compute the timesteps to process
    if read_time_d : 
        time_spec_d = get_time_spec_dictionary(control_d['time_spec'], master_d['time_d'])
    else :
        time_spec_d = get_time_spec_dictionary(control_d['time_spec'])
    print ( now(), 'report_on_model_run.py: time_spec_d = ')
    Core_Util.tree_print( time_spec_d )

    # levels to process 
    level_spec_d = Core_Util.get_spec_dictionary( control_d['level_spec'] )
    print ( now(), 'report_on_model_run.py: level_spec_d = ')
    Core_Util.tree_print( level_spec_d )

    # Main looping, first over times, then sections, then levels

    # Loop over times
    for i, time in enumerate( time_spec_d['time_list'] ) :
 
        age_Ma = time_spec_d['age_Ma'][i]
        runtime_Myr = time_spec_d['runtime_Myr'][i]

        # for each value on the list, remove units, and create int value
        time = time.replace('Ma', '')
        time = time.replace('Myr', '')
        time = int(time)

        # empty file_data
        file_data = []

        # cache for the file_format
        file_format_cache = ''

        # Loop over sections (fields) 
        for s in control_d['_SECTIONS_'] :

                # check for required parameter 'field'
                if not 'field' in control_d[s] :
                   print('ERROR: Required parameter "field" missing from section.')
                   print('       Skipping this section.')
                   continue # to next section
 
                # get the field name 
                field_name = control_d[s]['field']

                print('')
                print( now(), 'report_on_model_run.py: Processing: field =', field_name) 

                # get the data file name specifics for this field 
                file_name_component = field_to_file_map[field_name]['file']
                print( now(), 'report_on_model_run.py: file_name_component = ', file_name_component )

                # get the data file column name specifics for this field 
                field_column = field_to_file_map[field_name]['column']
                print( now(), 'report_on_model_run.py: field_column = ', field_column )

                # Loop over levels 
                for level in level_spec_d['list'] :

                    # ensure level is an int value 
                    level = int(level)
                    depth = int(depth_list[level])

                    #print( now(), '------------------------------------------------------------------------------')
                    print( now(), 'report_on_model_run.py: summary for', s, ': time =', time, '; age =', age_Ma, '; runtime_Myr =', runtime_Myr, '; level =', level, '; depth =', depth, ' km; field_name =', field_name)
                    #print( now(), '------------------------------------------------------------------------------')
                # end of loop over levels 

            # end of loop over sections

    # end of loop over times
#=====================================================================
#=====================================================================
#=====================================================================
def create_4DPlates_glb_file(CASE, FIELD, glb_filename, grid_list):
    '''From a list of [grid,age] tuples [(gridfile_name, age_Ma)], create a 4DPlates .glb file'''

    # start up the file 
    print( Core_Util.now(), 'create_4DPlates_glb_file:')
    print( Core_Util.now(), 'writing:', glb_filename )
    out_file = open( glb_filename, 'w' )

    # Header template
    header = '''<GLM filetype="Build project" version="2.3.10">
    <BuildProject2 id="1">
        <field_label>%(FIELD)s</field_label>
        <field_id>%(FIELD)s</field_id>
        <category>Citcoms/%(CASE)s</category>
        <predef_layer>true</predef_layer>
        <single_layer>false</single_layer>
        <use_age_filter>true</use_age_filter>
    </BuildProject2>'''

    section = '''
        <BuildFile2 parent="1">
            <filename source="user">%(G)s</filename>
            <field_label>%(AGE)s</field_label>
            <field_aod>%(AOD)d</field_aod>
            <field_aoa>%(AOA)d</field_aoa>
        </BuildFile2>\n'''

    footer = '''\n</GLM>\n'''

    if verbose: print ( grid_list )

    h = header % vars() 
    out_file.write( h )

    # first get lists of ages to ensure no under- or overlap
    AOD_list = []
    AOA_list = []

    # populate the age of disapearance with the grid's age
    for item in grid_list :
        AGE = float( item[1] )
        AOD_list.append( AGE )
    # populate the age of appearance with the prior grid's age
    for item in grid_list[1:] :
        AGE = float( item[1] )
        AOA_list.append( AGE )
    # complete the list with a final fake value
    AOA_list.append( 500 )
    
    for n, item in enumerate ( grid_list ) :
        G = item[0]
        AGE = float( item[1] )
        AOA = AOA_list[n] 
        AOD = AOD_list[n] 
        if AOD <= 0: AOD = -1

        print( Core_Util.now(), 'adding: grid:', G, 'AGE=', AGE, 'AOD=', AOD, '; AOA=', AOA )
        s = section % vars() 
        out_file.write( s )

    out_file.write( footer )
    out_file.close()
#=====================================================================
#=====================================================================
#=====================================================================
def dash_glb():
    '''From a list of standard grid file names as strings, generate a .glb file'''
    file_list = sys.argv[2:]
    file_name = file_list[0]

    # get case and field from a typical file name 
    case = file_name.split('-')[0]
    field = file_name.split('-')[1]

    print( now(), 'case=', case, '; field=', field)

    # generate the tuple list
    file_age_pairs = []
    for f in file_list:
        age = f.split('-')[2].replace('Ma','')
        file_age_pairs.append( [f,age] )

    # create the file 
    create_4DPlates_glb_file(case, field, case + '.glb', file_age_pairs)
#=====================================================================
#=====================================================================
def dash_levelstack():
    '''From a list of standard grid file names as strings, generate a .nc file'''
    file_list = sys.argv[2:]

    # NOTE: this function assumes the depths are sorted by zero padded values
    file_name = file_list[-1]

    # get case and field from a typical file name 
    case  = file_name.split('-')[0]
    field = file_name.split('-')[1]
    age   = file_name.split('-')[2]
    depth = file_name.split('-')[3]

    print( now(), 'case=', case, '; field=', field, '; age=', age)

    # generate the tuple list
    file_depth_pairs = []
    for f in file_list:
        d = f.split('-')[3].replace('km','')
        d = d.replace('.grd','')
        # make the depths negative for vis programs going into the earth
        d = -int(d)
        file_depth_pairs.append( [f,d] )

    # create the stacked file 
    nc_filename = case + '-' + field + '-' + age + '-levelstack' + '.nc'
    create_level_stack_nc(case, field, nc_filename, file_depth_pairs)
#=====================================================================
#=====================================================================
def create_level_stack_nc(CASE, FIELD, nc_filename, grid_list):
    '''From a list of [grid,depth] tuples [(gridfile_name, depth_km)], create a single .nc'''

    # start up the file 
    print( Core_Util.now(), 'create_level_stack_nc:')

    # a list of strings to hold all the data combined
    stack_list = []

    # Get the list of depths as ints;
    depth_list = []
    for n, item in enumerate ( grid_list ) :
        depth_list.append( item[1] )

    print( Core_Util.now(), 'create_level_stack_nc: depth_list =', depth_list)
    print( Core_Util.now(), 'create_level_stack_nc: sorted(depth_list) =', sorted(depth_list) )

    # get the number of depths
    d = len(grid_list)
    d_min = sorted(depth_list)[0]
    d_max = sorted(depth_list)[-1]

    # open a file to get common header 
    cmd = 'ncdump ' + grid_list[-1][0]
    s = subprocess.check_output( cmd, shell=True, universal_newlines=True)
    l = s.split('z =')
    h = l[0]

    h_lines = h.split('\n')

    # add the depth dimension
    n = h_lines.index('dimensions:')
    depth_dim = '\tdepth = ' + str(d) + ' ;'
    h_lines.insert(n+1, depth_dim)

    # Add the depth variable
    depth_var = '\tfloat depth(depth) ;\n'          + \
                  '\t\tdepth:units = "kilometer" ;\n' + \
                  '\t\tdepth:positive = "up" ;\n'     + \
                  '\t\tdepth:actual_range = %(d_min)s., %(d_max)s. ;' % vars() 
    n = h_lines.index('variables:')
    h_lines.insert(n+1, depth_var)

    # Add the values for depth
    n = h_lines.index('data:')
    depth_data = ' depth = '
    depth_data = depth_data + ', '.join(str(x) for x in depth_list) + ';'
    h_lines.insert(n+1, depth_data)

    # Change the z variable to FIELD
    # NOTE Loop over lines because we don't know what the actual_range will be 

    for n, line in enumerate( h_lines ) :
        if 'float z(lat, lon) ;' in line :
            h_lines[n] = '\tfloat ' + FIELD + '(depth, lat, lon) ;'
            break # out of loop
        
    for n, line in enumerate( h_lines ) :
        if 'z:long_name = "%(FIELD)s" ;' % vars() in line: 
            h_lines[n] = '\t\t%(FIELD)s:long_name = "%(FIELD)s" ;' % vars() 
            break # out of loop

    for n, line in enumerate( h_lines ) :
        if 'z:_FillValue = NaNf ;' in line: 
            h_lines[n] = '\t\t%(FIELD)s:_FillValue = NaNf ;' % vars() 
            break # out of loop

    for n, line in enumerate( h_lines ) :
       if 'z:actual_range' in line:
           h_lines[n] = '\t\t%(FIELD)s:actual_range = 0., 1. ;' % vars()
           break # out of loop
     
    # add the correct field id to the header block
    h_lines.append(' %(FIELD)s =' % vars() )

    # reassemble to a big string
    header = '\n'.join(h_lines)

    # report 
    #print( 'header = ', header ) 
    #return

    # append the header string
    stack_list.append(header)

    # get the number of files to process 
    num = len(grid_list)

    # loop over all grids 
    for n, item in enumerate ( grid_list ) :
        G = item[0]
        depth = -item[1]
        logging.debug(f'processing: n={n}, grid={G}, depth={depth}')

        # dump the data to ascii
        cmd = 'ncdump ' + G
        #print( now(), 'cmd =', cmd)
        s = subprocess.check_output( cmd, shell=True, universal_newlines=True)
        l = s.split('z =')
        data = l[1]
        #print(data)

        # the last file in the list is special 
        if not n == num -1 :

            # change the ';' to ','
            data = data.replace(';', ',')
            # remove the final '}'
            data = data.replace('}', '')
       
        # append the data block 
        stack_list.append( data )

    # assemble stack
    stack_string = ''.join(stack_list)
    #print(stack_string)

    # write out the stack as a cdl file 
    cdl_filename = nc_filename.replace('.nc', '.cdl')
    print(Core_Util.now(), 'create_level_stack_nc: write:', cdl_filename)
    cdl_outfile = open(cdl_filename, 'w')
    cdl_outfile.write( stack_string )
    cdl_outfile.close()

    # write a .ncml file 
    ncml_base = '''<?xml version="1.0" encoding="UTF-8"?>
<netcdf xmlns="http://www.unidata.ucar.edu/namespaces/netcdf/ncml-2.2" location="file:./%(NC)s">

  <variable name="time">
    <attribute name="units" value="years since 1-01-01 00:00:00" />
  </variable>
  <variable name="%(FIELD)s">
    <attribute name="Coordinates" value="lat lon depth time" />
  </variable>

</netcdf>'''
    NC = nc_filename
    ncml = ncml_base % vars()

    # create the ncml file 
    ncml_filename = nc_filename.replace('.nc', '.ncml')

    print(Core_Util.now(), 'create_level_stack_nc: write:', ncml_filename)
    ncml_outfile = open(ncml_filename, 'w')
    ncml_outfile.write(ncml)
    ncml_outfile.close()

    # create the final output 
    cmd = 'ncgen -o ' + nc_filename + ' ' + cdl_filename
    print(Core_Util.now(), 'create_level_stack_nc: cmd=', cmd)
    s = subprocess.check_output( cmd, shell=True, universal_newlines=True)
    print(s)
    
#=====================================================================
#=====================================================================
def test_time_spec(control_d, master_d):
    ''' test the time specs processing '''

    print('test_time_spec:')

    # show the time dict
    #print('\n', Core_Util.now(), 'master_d[\'time\'] = ')
    #Core_Util.tree_print(master_d['time'])

    for section in control_d['_SECTIONS_']:

        print( Core_Util.now(), '---------------------------------------------------------------------------------')
        print(section, ' has time_spec lists: ')
        # get the time_spec string
        time_spec = control_d[section]['time_spec']

        # test without the time_d  ( default argument is FALSE )
        time_spec_d = get_time_spec_dictionary(time_spec)

        print(Core_Util.now(), 'NO TIME_D:')
        print(Core_Util.now(), 'time_spec_d = ')
        Core_Util.tree_print(time_spec_d)

        # get the time_spec dictionary 
        time_spec_d = get_time_spec_dictionary(time_spec, master_d['time_d'])

        print(Core_Util.now(), 'WITH TIME_D:')
        print(Core_Util.now(), 'time_spec_d = ')
        Core_Util.tree_print(time_spec_d)
#=====================================================================
#=====================================================================
#=====================================================================
def test():
    ''' This is the main test function ''' 
    print("This is the main test function")
    print("len(sys.argv) = ", len(sys.argv) )
    print("sys.argv = ", sys.argv )
    print('field_to_file_map=')
    Core_Util.tree_print(field_to_file_map)

    # show/hide some of the messages from other Core_* modules
    Core_Util.verbose = False

    # create an empty dictionary to hold all the main data
    control_d = {}

    # get the .cfg file as a dictionary
    control_d = Core_Util.parse_configuration_file( sys.argv[1] )

    # set the pid file 
    pid_file = control_d['pid_file']

    # get the master dictionary
    master_d = get_all_pid_data( pid_file )

    # show the dict
    #print('\nCore_Citcom.test()\n', Core_Util.now(), 'master_d = ')
    #Core_Util.tree_print(master_d)

    # use the dict 
    test_time_spec(control_d, master_d)

#=====================================================================
def test_find_times():
    '''a test function to call the find_available_timestep_from_* functions.
Conventions used:
sys.argv[1] = "-t"
sys.argv[2] = "find_times"
sys.argv[3] = pidXXXX.cfg file 
sys.argv[4] = time value 
sys.argv[5] = "time_step" or "age"
'''
    print(now(), 'test_find_times: START')

    # error check
    if len(sys.argv) != 6:
       print(now(), 'Core_Citcom.test_find_times(): sys.argv must have these values set:')
       print(now(), 'sys.argv[1] = "-t"')
       print(now(), 'sys.argv[2] = "find_times"')
       print(now(), 'sys.argv[3] = pidXXXX.cfg file of case to test ')
       print(now(), 'sys.argv[4] = time value to find ')
       print(now(), 'sys.argv[5] = "time_step" or "age"')
       sys.exit(-1)

    # get the master_d 
    master_d = get_all_pid_data( sys.argv[3] )

    # get the requested time value
    request_t = int( sys.argv[4] )

    # determine which specific function to call 
    if sys.argv[5] == 'time_step':
        print(now(), 'test_find_times: test find_available_timestep_from_timestep:', request_t)
        find_available_timestep_from_timestep( master_d, 'temp', request_t)
        print() 

    elif sys.argv[5] == 'age':
        print(now(), 'test_find_times: test find_available_timestep_from_age:', request_t)
        find_available_timestep_from_age( master_d, 'temp', request_t)
        print() 

    else:
        print(now(), 'test_find_times: arg[5] must be one of "time_step" or "age"')
    
#=====================================================================
#=====================================================================
def test_function():
    ''' a handler function to call specific test functions for this module'''

    # NOTE: args[1] will be '-t' 
    # use args[2] to determine which specific test to call

    global verbose
    verbose = True

    print( Core_Util.now(), 'Core_Citcom: main test_function:')
    print() 

    # Check args for specific test calls 
    if sys.argv[2] == 'find_times':
        test_find_times()

    # Add more calls to specific test functions here as needed
    # ... 

#=====================================================================
#=====================================================================
if __name__ == "__main__":
    import Core_Citcom

    if len(sys.argv) > 1:

        # make the example configuration file
        if sys.argv[1] == '-e' :
            #make_example_configuration_file()
            sys.exit(0)

        # run a specific test on sys.argv
        if sys.argv[1] == '-t':
            test_function()
            sys.exit(0)

        # Create a .glb file from a set of file names given on the command line argv[2:]
        if sys.argv[1] == '-glb':
            dash_glb()

        if sys.argv[1] == '-levelstack':
            dash_levelstack()
    else:
        help(Core_Citcom)
#=====================================================================
#=====================================================================
#=====================================================================
    # FIXME : Save for another test function 

    # Test print of the restart params dictionaries 
    # 
    #print( Core_Util.now(), 'Core_Citcom: test_function: total_topography_restart_params =')
    #Core_Util.tree_print ( total_topography_restart_params )
    #print( Core_Util.now() )
    #print( Core_Util.now(), 'Core_Citcom: test_function: dynamic_topography_restart_params =')
    #Core_Util.tree_print ( dynamic_topography_restart_params )

    # 
    # Test of dimensionalizing a grid 
    # 
    #print( Core_Util.now(), 'Core_Citcom: test_function: field_to_dimensional_map =')
    #Core_Util.tree_print ( field_to_dimensional_map )
    #dimensionalize_grid( 'BAD_FIELD', sys.argv[2] , sys.argv[2].replace('.grd', '.dimensional.grd') )
    #dimensionalize_grid( 'temp', sys.argv[2] , sys.argv[2].replace('.grd', '.dimensional.grd') )

    # Test reporting of model run params 
    #report_on_model_run( sys.argv[2] )
#=====================================================================
#=====================================================================
#=====================================================================
