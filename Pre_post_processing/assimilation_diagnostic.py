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
# Last Update: Dan J. Bower, 15th November 2013
#=====================================================================

import subprocess, sys
from Core_GMT import apos, callgmt
from Core_Util import now
import numpy as np
from scipy.integrate import simps
from scipy.special import erf
import Core_Util, Core_Citcom, Core_GMT
# this function should probably be generalized and moved into a Core_
# module
from make_history_for_age import write_coordinates_by_cap

# global variables
#====================================================================
verbose = True
#====================================================================
#====================================================================
#====================================================================
def usage():
    """print usage message and exit"""

    print( now(), '''usage: assimilation_diagnostic.py [-e] configuration_file.cfg

options and arguments:

-e: if the optional -e argument is given this script will print to
standard out an example configuration control file.  The parameter 
values in the example configuration_file.cfg file may need to be 
edited or commented out depending on intended use.
''')

    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
def main():
    """main sequence of script actions"""

    print( now(), 'assimilation_diagnostic.py:' )
    print( now(), 'main:' )

    # get the .cfg file as a dictionary
    control_d = Core_Util.parse_configuration_file( sys.argv[1] )

    PLOT_CROSS_SECTIONS = control_d['PLOT_CROSS_SECTIONS']
    PLOT_MAPS = control_d['PLOT_MAPS']

    # get the master dictionary
    master_d = Core_Citcom.get_all_pid_data( control_d['pid_file'] )

    # get times to process and plot
    time_d = Core_Citcom.get_time_spec_dictionary( control_d['time_spec'], master_d['time_d'])
    master_d['control_d'] = control_d
    # N.B. master_d['time_d'] contains all the time info from citcoms model
    # N.B. master_d['control_d']['time_d'] contains specific times to process
    master_d['control_d']['time_d'] = time_d

    # func_d is a separate dictionary that is used to transport
    # temporary files and objects between functions
    master_d['func_d'] = {'rm_list': []}

    # find track locations
    make_profile_track_files( master_d )

    make_cpts( master_d )

    # make cross-sections
    if PLOT_CROSS_SECTIONS:
        make_cross_section_diagnostics( master_d )

    # make maps
    if PLOT_MAPS:
        ps_l = []
        for tt in range(len(time_d['time_list'])):
            ps = make_map_postscript( master_d, tt )
            ps_l.append( ps )
        pdf_name = control_d['prefix'] + '_'
        pdf_name += 'Map.pdf'
        Core_Util.make_pdf_from_ps_list( ps_l, pdf_name )

    # clean up
    Core_Util.remove_files( master_d['func_d']['rm_list'] )

#====================================================================
#====================================================================
#====================================================================
def make_cross_section_diagnostics( master_d ):

    # parameters from dictionary
    control_d = master_d['control_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']
    pid_d = master_d['pid_d']
    datafile = pid_d['datafile']
    depth_km = master_d['coor_d']['depth_km']
    FILE_VBCS = int( pid_d['file_vbcs'] )
    FULL_SPHERE = pid_d['FULL_SPHERE']
    LITH_AGE = int( pid_d['lith_age'])
    LITH_AGE_TIME = int( pid_d['lith_age_time'])
    rad_in = str(pid_d['radius_inner'])
    rad_out = str(pid_d['radius_outer'])
    radius = master_d['coor_d']['radius']
    REGIONAL = pid_d['REGIONAL']
    rm_list = func_d['rm_list']
    scalet = pid_d['scalet']
    time_d = master_d['control_d']['time_d']
    velocity_scale = control_d['velocity_scale']

    # hard-coded parameters
    blockmedian_I = '0.2/0.0035' # deg/non-dim radius
    surface_I = '0.1/0.00175' # deg/non-dim radius
    blockmedianc_I = '25/22' # km/km
    # for coarse plots use:
    #surfacec_I = '12.5/11' # km/km
    # for smoothed plots use:
    # 5/3 sometimes
    # 10/8
    surfacec_I = '5/3' # km/km
    surface_T = '0.25'
    master_X = 0.5 # X offset from margin for plot
    master_Y = 0.5 # Y offset from margin for plot
    x_cart_min = 0
    z_cart_min = 0
    z_cart_max = pid_d['radius_km']*(1-pid_d['radius_inner'])

    # for small Cartesian plots (plotting only?)
    X_scart_l = [0.2, 2.75, 5.3, 0.2, 2.75, 5.3]
    # XXX DJB - push over for larger temp plot (Laramide flat slab)
    #X_scart_l = [0.4, 3.95, 5.5, 0.4, 2.95, 5.5]
    #X_scart_l = [0.4, 2.95, 5.5, 0.4, 2.95, 5.5]
    Y_scart_l = [5.7, 5.7, 5.7, 2.9, 2.9, 2.9]
    B_scart_l = ['a0.5f0.25', 'a1', 'a1', 'a5f2.5', 'a5f2.5', 'a1']
    title_scart_l = ['''Temp (T')''', '|Velocity| (cm/yr)', '''Viscosity (@~h@~')''', 'v@-x@- (cm/yr)', 'v@-z@- (cm/yr)', 'log10(|divv/velocity|)']

    field_scart_l = ['temp', 'vmag', 'visc', 'tangent', 'radial', 'divv_norm']

    # field list for generating data
    field_l = ['sten_temp', 'temp', 'divv_norm', 'visc'] # 'pressure'] #, 'visc']
    # use this next line for kinematic bcs only (no slab stencil)
    #field_l = ['temp', 'divv_norm', 'visc'] # 'pressure'] #, 'visc']
    gpfx = 'grid/' + datafile + '.'
    vtpr_grid_files_l = []

    # loop over sections (fields)
    for ss, section in enumerate( control_d['_SECTIONS_']):

        ps_l = [] # to store postscripts for this section

        # section dictionary
        section_d = control_d[section]
        lon0 = section_d['lon0']
        lat0 = section_d['lat0']
        lon1 = section_d['lon1']
        lat1 = section_d['lat1']

        letter = section.split('_')[1]

        proj_name_l = [func_d['annular_project'][ss]]
        proj_name_l.append( func_d['rectangular_project'][ss] )
        print( proj_name_l )

        # make velocity xyzs
        velo_ann_name = 'velocity_' + section + '.xyz'
        velo_cart_name = 'velocity_' + section + '.c.xyz'
        tang_ann_name = 'tangent_' + section + '.xyz'
        rad_ann_name = 'radial_' + section + '.xyz'
        vmag_ann_name = 'vmag_' + section + '.xyz'
        rm_list.extend( [velo_ann_name, velo_cart_name,
            tang_ann_name, rad_ann_name, vmag_ann_name] )

        # *** loop over times ***
        for tt, time in enumerate( time_d['time_list'] ):

            # parameters
            runtime_Myr = time_d['runtime_Myr'][tt]
            age_Ma = time_d['age_Ma'][tt]

            # make grid name lists for tracking
            grid_d = { field+'_list': [] for field in field_l } # reset
            vtpr_grid_files_l = [] # reset
            # build lists of grid file names for each field
            for depth in depth_km:
                gsfx = '.' + str(int(depth)) + '.' + str(time) + '.grd'
                # velocity
                velo_t = [gpfx + velo + gsfx for velo in ['vx','vy','vz']]
                vtpr_grid_files_l.append( tuple(velo_t) )
                # other fields
                for field in field_l:
                    grid_d[field+'_list'].append( gpfx + field + gsfx )

            # rectangle velocity (exclude polar angular offset)
            Core_Util.make_rectangle_cross_section_velocity( master_d,
                proj_name_l[0], velocity_scale, vtpr_grid_files_l,
                [velo_cart_name, '', '', ''] )
            phi_deg, rad, val0, val1 = np.loadtxt( velo_cart_name, unpack=True )
            x = np.radians(phi_deg) * pid_d['radius_km']
            z = (1-rad) * pid_d['radius_km']
            np.savetxt( velo_cart_name, np.column_stack((x,z,val0,val1)) )

            # annulus velocity (include polar angular offset)
            out_names = [velo_ann_name, tang_ann_name, rad_ann_name,
                vmag_ann_name]
            Core_Util.make_annulus_cross_section_velocity( master_d,
                proj_name_l[0], velocity_scale, vtpr_grid_files_l, 
                out_names )

            # make other xyzs (excluding velocity)
            for field in field_l:
                name = field + '_' + section + '.xyz'
                rm_list.append( name )
                pao, x_ann_max = Core_Util.make_annulus_xyz( master_d,
                    proj_name_l[0], name, grid_d[field+'_list'] )

            # idealized lithosphere temperature overlay (from age grids)
            xyz = 'lithtemp_' + section + '.xyz'
            rm_list.append( xyz )
            age_str = '%(age_Ma)d' % vars()
            grid = control_d['lith_thermal_age_grids'] + age_str + '.grd'
            Core_Util.find_value_on_line( proj_name_l[0], grid, xyz )
            lithlon, lithlat, lithdist1, lithage_Ma = np.loadtxt( xyz, unpack=True )
            lithdist = np.tile( lithdist1, pid_d['nodez'] )
            lithage_Ma = np.tile( lithage_Ma, pid_d['nodez'] )
            lithrad = []
            for rad in radius:
                lithrad.extend( [rad for xx in range( len(lithdist1) )] )
            lithrad = np.array( lithrad )
            lithtemp = erf((1.0-lithrad)/(2.0*np.sqrt(lithage_Ma/scalet)))
            np.savetxt( xyz, np.column_stack( (lithdist, lithrad, lithtemp) ) ) 

            # make grids
            field_l2 = list( field_l )
            field_l2.extend( ['tangent', 'radial', 'vmag', 'lithtemp'] )
            for field in field_l2:
                # annulus
                xyz_name = field + '_' + section + '.xyz'
                block_name, grid_name = get_bm_and_grid_file_names( xyz_name )
                rm_list.extend( [block_name, grid_name] )
                R_ann = '0/' +str( x_ann_max ) + '/' + rad_in + '/' + rad_out
                cmd = xyz_name + ' -I' + blockmedian_I + ' -R' + R_ann
                callgmt( 'blockmedian', cmd, '', '>', block_name )
                cmd = block_name + ' -I' + surface_I + ' -R' + R_ann
                cmd += ' -T' + surface_T
                callgmt( 'surface', cmd, '', '', '-G' + grid_name )

                # Cartesian
                xyz_cart_name = field + '_' + section + '.c.xyz'
                block_cart_name, grid_cart_name = \
                    get_bm_and_grid_file_names( xyz_cart_name )
                rm_list.extend( [xyz_cart_name, block_cart_name,
                    grid_cart_name] )
                phi_deg, rad, val = np.loadtxt( xyz_name, unpack=True )
                x_cart = np.radians(phi_deg) * pid_d['radius_km']
                x_cart_max = np.max( x_cart )
                z_cart = (1-rad)*pid_d['radius_km']
                np.savetxt( xyz_cart_name, 
                    np.column_stack((x_cart, z_cart, val)) )
                R_cart = '%(x_cart_min)s/%(x_cart_max)s/' % vars()
                R_cart += '%(z_cart_min)s/%(z_cart_max)s' % vars()
                cmd = xyz_cart_name + ' -I' + blockmedianc_I + ' -R' + R_cart 
                callgmt( 'blockmedian', cmd, '', '>', block_cart_name )
                cmd = block_cart_name + ' -I' + surfacec_I + ' -R' + R_cart 
                cmd += ' -T' + surface_T
                callgmt( 'surface', cmd, '', '', '-G' + grid_cart_name )

            # filter stencil grid
            #grid = 'sten_temp_' + section + '.c.grd'
            #filter_width = 100.0
            #cmd = grid + ' -D0 -Fg%(filter_width)g -V' % vars()
            #callgmt( 'grdfilter', cmd, '', '', '-G' + grid )

            # ----------------
            # initialize
            # ----------------
            # moved higher in script
            gmtset()

            ps = control_d['prefix']
            ps += '_' + section
            ps += '_%(lon0)s_%(lat0)s_%(lon1)s_%(lat1)s_' % vars()
            ps += str(time) + '.ps'
            ps_l.append( ps )
            opts_d = Core_GMT.start_postscript( ps )

            # text
            # build option dictionary for pstext commands
            textopts_d = opts_d.copy()
            textopts_d['J'] = 'x1'
            textopts_d['R'] = '0/8.5/0/11'
            textopts_d['X'] = apos(master_X)
            textopts_d['Y'] = apos(master_Y)
            stdin = '0 10.0 16 0 4 ML %s\n' % section
            stdin += '1.5 10.0 16 0 4 ML %s\n' % datafile
            stdin += '2.5 10.0 16 0 4 ML Timestep %s\n' % time
            stdin += '4.5 10.0 16 0 4 ML Runtime %0.1f Myr\n' % runtime_Myr
            stdin += '6.5 10.0 16 0 4 ML Age %0.1f Ma\n' % age_Ma
            stdin += 'EOF'
            callgmt( 'pstext', '', textopts_d, '<< EOF >>', ps + '\n' + stdin )


            # *** annulus ***
            # annopts_d for annular plots
            # temperature annulus
            annopts_d = opts_d.copy()
            annopts_d['B'] = 'a20/0.1:"Radius":WS'
            annopts_d['C'] = 'temp.cpt'
            annopts_d['J'] = 'pa3.5/' + str(pao) # used to have Pa3.5
            annopts_d['R'] = R_ann
            annopts_d['X'] = apos( master_X )
            annopts_d['Y'] = apos( master_Y + 7.8 )
            grid = 'temp_' + section + '.grd'
            callgmt( 'grdimage', grid, annopts_d, '>>', ps )
            del annopts_d['B']
            del annopts_d['C']
            # velocity annulus overlay
            velo_sample_name = velo_ann_name.rstrip('xyz') + 'subsample.xyz'
            rm_list.append( velo_sample_name )
            # XXX was 11 not 15
            Core_Util.sample_regular_velocity_from_file( velo_ann_name, 
                velo_sample_name, 15, pid_d['nodez'], 9 ) # XXX
            annopts_d['G'] = 'black'
            annopts_d['S'] = 'V0.015i/0.06i/0.05i'
            callgmt( 'psxy', velo_sample_name, annopts_d, '>>', ps )
            del annopts_d['G']
            del annopts_d['S']
            # mark profile orientation (X to X')
            annopts_d['N'] = ' '        
            text = '0 1.05 14 0 4 MC %(letter)s\n' % vars()
            text += '''%(x_ann_max)s 1.05 14 0 4 MC %(letter)s'\n''' % vars()
            text += 'EOF'
            # XXX DJB uncomment line below
            callgmt( 'pstext', '', annopts_d, '<< EOF >>', ps + '\n' + text )
            del annopts_d # delete dictionary


            # *** surface velocity ***
            # bvelopts_d for surface velocity (usually imposed)
            bvelopts_d = opts_d.copy()
            xy = 'bvel_' + section + '.c.xy'
            rm_list.append( xy )
            grid = 'tangent_' + section + '.c.grd'
            Core_Util.find_value_on_line( proj_name_l[1], grid, xy )
            x_bvel, y_bvel, velo_bvel = np.loadtxt( xy, unpack=True )
            np.savetxt( xy, np.column_stack( (x_bvel, velo_bvel) ) )
            bvelopts_d['R'] = '0/%(x_cart_max)s/-5/7.5' % vars()
            bvelopts_d['J'] = 'X3.5/0.5'
            bvelopts_d['B'] = 'f1000/a5f2.5:"v@-x@- (cm/yr)":Wes'
            bvelopts_d['W'] = '4,black'
            bvelopts_d['X'] = apos( master_X + 4.1 )
            bvelopts_d['Y'] = apos( master_Y + 9.2 )
            callgmt( 'psxy', xy, bvelopts_d, '>>', ps )
            del bvelopts_d # delete dictionary
            # maximum gradient in surface velocity can be used to
            # locate the position of the trench
            # used to orient small Cartesian plots and profiles
            grad_bvel = np.abs( np.gradient( velo_bvel ) )
            grad_bvel = grad_bvel[10:len(grad_bvel)-10] # remove tapering at edges
            x_trench = x_bvel[ grad_bvel.argmax()+10 ]
            # XXX DJB
            # XXX DJB 2250 for Laramide flat slab
            #x_trench = 2750
            #x_trench = 400

            # *** buoyancy profiles and integrals ***
            slab_out_l, slab_int_a, rm_list = slab_buoyancy_profiles( control_d,
                section, section_d, x_trench, rm_list )
            lith_out_l, lith_int_a = lith_buoyancy_profiles( control_d,
                section, x_trench )
            rm_list.extend( [slab_out_l, lith_out_l] )


            # flip polarity of radial velocity so positive velocities
            # are now down (looks more natural in slab plots)
            # this operation must be AFTER the computation of the
            # velocity vectors because the velocity vector routines
            # require radial velocity with outward normal.
            grid = 'radial_' + section + '.c.grd'
            callgmt( 'grdmath', grid + ' -1 MUL', '', '=', grid )


            # *** Cartesian ***
            # temperature Cartesian
            cartopts_d = opts_d.copy()
            cartopts_d['B'] = 'a1000f500:"x (km)":/f50a100:"Depth (km)":WeSn'
            #cartopts_d['B'] = 'a1000f500:"x (km)":/f100a200:"Depth (km)":WeSn'
            cartopts_d['J'] = 'X3.5/-1'
            # XXX DJB - 200 or 500
            cartopts_d['R'] = '0/%(x_cart_max)s/0/200' % vars()
            cartopts_d['X'] = apos( master_X + 4.1 )
            cartopts_d['Y'] = apos( master_Y + 8.0 )
            callgmt( 'psbasemap', '', cartopts_d, '>>', ps )
            del cartopts_d['B']
            cartopts_d['C'] = 'temp.cpt'
            grid = 'temp_' + section + '.c.grd'
            callgmt( 'grdimage', grid, cartopts_d, '>>', ps )
            del cartopts_d['C']
            # idealized lithosphere temperature overlay (from age grids)
            grid = 'lithtemp_' + section + '.c.grd'
            cartopts_d['C'] = 'agetemp.cont'
            cartopts_d['W'] = '3,black'
            callgmt( 'grdcontour', grid, cartopts_d, '>>', ps )
            del cartopts_d['C']
            del cartopts_d['W']
            # lithosphere assimilation depth overlay (constant depth)
            xy = 'lithdepth_' + section + '.c.xy'
            rm_list.append( xy )
            lith_age_depth = pid_d['lith_age_depth']
            lith_depth = np.tile( lith_age_depth*pid_d['radius_km'], len( x_cart ) )
            np.savetxt( xy, np.column_stack( (x_cart, lith_depth) ) )
            cartopts_d['W'] = '3,red'
            # XXX DJB uncomment line below to plot line
            # callgmt( 'psxy', xy, cartopts_d, '>>', ps )
            del cartopts_d # delete dictionary


            # *** small Cartesian ***
            # automagically determine bounds (R)
            lith_profile_distance = float( control_d['lith_profile_distance'] )
            x_scart_min = x_trench - lith_profile_distance - 300
            # XXX changed to 2500 from 2000
            # XXX DJB - 2636.0 for Laramide
            # XXX DJB - 2696 for IBM
            # 2000 for subduction initiation
            # 2500 for example case
            fudge = 2000
            x_scart_max = x_scart_min + fudge
            if x_scart_max > x_cart_max:
                x_scart_max = x_cart_max
                x_scart_min = x_cart_max - fudge

            for ff, field in enumerate( field_scart_l ):
                title = title_scart_l[ff]
                scartopts_d = opts_d.copy()
                scartopts_d['B'] = 'a500f250:"x (km)":/f250a500:"Depth (km)":."%(title)s"::' % vars()
                scartopts_d['J'] = 'x0.001/-0.001'
                # XXX DJB - 1500 or 750
                scartopts_d['R'] = '%(x_scart_min)s/%(x_scart_max)s/0/1500' % vars()
                scartopts_d['X'] = apos( master_X + X_scart_l[ff] )
                scartopts_d['Y'] = apos( master_Y + Y_scart_l[ff] )
                if ff == 0 or ff == 3: scartopts_d['B'] += 'WSn'
                else: scartopts_d['B'] += 'wSn'
                callgmt( 'psbasemap', '', scartopts_d, '>>', ps )
                del scartopts_d['B']

                #if field != 'velocity':
                grid = field + '_' + section + '.c.grd'
                scartopts_d['C'] = field + '.cpt'
                callgmt( 'grdimage', grid, scartopts_d, '>>', ps )
                # new dictionary for psscale
                psscaleopts_d = scartopts_d.copy()
                psscaleopts_d['B'] = B_scart_l[ff]
                psscaleopts_d['D'] = '1.0/-0.5/2/0.15h' 
                del psscaleopts_d['J']
                del psscaleopts_d['R']
                callgmt( 'psscale', '', psscaleopts_d, '>>', ps )
                del scartopts_d['C']
                #else:
                if field == 'vmag' or field == 'temp':
                   # velocity annulus overlay
                    velo_sample_name = velo_cart_name.rstrip('xyz') + 'subsample.xyz'
                    rm_list.append( velo_sample_name )
                    Core_Util.sample_regular_velocity_from_file(
                        velo_cart_name, velo_sample_name, 8, pid_d['nodez'], 7 ) # XXX
                    scartopts_d['G'] = 'black'
                    scartopts_d['S'] = 'V0.015i/0.06i/0.05i'
                    callgmt( 'psxy', velo_sample_name, scartopts_d, '>>', ps )
                    del scartopts_d['G']
                    del scartopts_d['S']

                    # plot velocity arrow
#                    vscaleopts_d = { 'R': '0/3/0/3', 'J': 'x1', 'K': ' ',
#                        'O': ' ' }
                    vscaleopts_d = { 'R': '0/10/0/10', 'J': 'x1', 'K': ' ',
                        'O': ' ' }
                    #vscaleopts_d['C'] = '0.05i'
                    #vscaleopts_d['D'] = '1.75/-0.35/0.875/0.20/MC'
                    vscaleopts_d['D'] = '0/1.9/0.875/0.20/MC'
                    vscaleopts_d['X'] = scartopts_d['X']
                    vscaleopts_d['Y'] = scartopts_d['Y']
                    #vscaleopts_d['G'] = 'black'
                    Core_GMT.plot_velocity_scale( vscaleopts_d, velocity_scale, \
                        0.25, ps )
                    del vscaleopts_d


                # thin black temperature contours on velocity plots
                if field in ['tangent', 'radial', 'vmag', 'visc' ]: # 'pressure']:
                    grid = 'temp_' + section +'.c.grd'
                    scartopts_d['C'] = 'agetemp.cont'
                    callgmt( 'grdcontour', grid, scartopts_d, '>>', ps )
                    del scartopts_d['C']

                # temperature stencil
                scartopts_d['C'] = 'sten_temp.cont'
                scartopts_d['W'] = '3,green'
                grid = 'sten_temp_' + section + '.c.grd'
                # XXX DJB do not plot stencil
                callgmt( 'grdcontour', grid, scartopts_d, '>>', ps )
                del scartopts_d['C']
                scartopts_d['W'] = '4,purple' # slab profile
                callgmt( 'psxy', slab_out_l[0], scartopts_d, '>>', ps )
                scartopts_d['W'] = '4,black' # lithosphere profile
                callgmt( 'psxy', lith_out_l[0], scartopts_d, '>>', ps )
                del scartopts_d['W']


            # temperature profiles
            profopts_d = opts_d.copy() # reset
            slab_profile_width = float( control_d['slab_profile_width'] )
            xy = 'slabp_' + section + '_temp.c.xy'
            profopts_d['R'] = '0/%s/0/1.05' % slab_profile_width
            profopts_d['J'] = 'X1/1'
            profopts_d['B'] = '''a200f100:"Profile dist, p (km)":/f0.25a0.5:"Temp, T'":WSn'''
            profopts_d['W'] = '4,purple'
            profopts_d['X'] = apos( master_X + 0.2 )
            profopts_d['Y'] = apos( master_Y + 0.5 )
            callgmt( 'psxy', xy, profopts_d, '>>', ps )
            del profopts_d['B']
            profopts_d['W'] = '4,black'
            xy = 'lithp_' + section + '_temp.c.xy'
            callgmt( 'psxy', xy, profopts_d, '>>', ps )



            # velocity profiles
            profopts_d = opts_d.copy() # reset
            xy = 'slabp_' + section + '_velo.c.xy'
            profopts_d['R'] = '0/%s/-0.5/7.5' % slab_profile_width
            profopts_d['J'] = 'X1/1'
            profopts_d['B'] = 'a200f100:"Profile dist, p (km)":/a5f2.5:"Velo, v (cm/yr)":WSn'
            profopts_d['W'] = '4,purple'
            profopts_d['X'] = apos( master_X + 1.9 )
            profopts_d['Y'] = apos( master_Y + 0.5 )
            callgmt( 'psxy', xy, profopts_d, '>>', ps )
            del profopts_d['B']
            profopts_d['W'] = '4,black'
            xy = 'lithp_' + section + '_tangent.c.xy'
            callgmt( 'psxy', xy, profopts_d, '>>', ps )



            # buoyancy flux profiles
            profopts_d = opts_d.copy() # reset
            xy = 'slabp_' + section + '_flux.c.xy'
            profopts_d['R'] = '0/%s/-0.5/6' % slab_profile_width
            profopts_d['J'] = 'X1/1'
            profopts_d['B'] = '''a200f100:"Profile dist, p (km)":/a5f2.5:"Flux, q (cm/yr)":WSn'''
            profopts_d['W'] = '4,purple'
            profopts_d['X'] = apos( master_X + 3.6 )
            profopts_d['Y'] = apos( master_Y + 0.5 )
            callgmt( 'psxy', xy, profopts_d, '>>', ps )
            del profopts_d['B']
            profopts_d['W'] = '4,black'
            xy = 'lithp_' + section + '_flux.c.xy'
            callgmt( 'psxy', xy, profopts_d, '>>', ps )

            # report (print) integral data
            print_integral_data( opts_d, master_X, master_Y, ps,
                slab_int_a, lith_int_a )

            # close
            Core_GMT.end_postscript( ps )

            # end loop over time

        # end loop over section
        pdf_name = control_d['prefix'] + '_'
        pdf_name += section + '.pdf'
        Core_Util.make_pdf_from_ps_list( ps_l, pdf_name )

#====================================================================
#====================================================================
#====================================================================
def make_profile_track_files( master_d ):

    '''Make annular and rectangular track files.'''

    if verbose: print( now(), 'make_profile_track_files:' )

    control_d = master_d['control_d']
    func_d = master_d['func_d']
    pid_d = master_d['pid_d']
    rm_list = func_d['rm_list']

    annular_project = []
    rectangular_project = []

    # loop over all sections
    for section in control_d['_SECTIONS_']:

        # dictionary for this section
        section_d = control_d[section]

        lon0 = section_d['lon0']
        lat0 = section_d['lat0']
        lon1 = section_d['lon1']
        lat1 = section_d['lat1']

        # annular
        proj_name = 'annular_project_' + section + '.xy'
        annular_project.append( proj_name )
        rm_list.append( proj_name )
        incr = 0.5 # sample every 0.5 degrees
        Core_Util.make_great_circle_with_two_points( lon0, lat0,
            lon1, lat1, incr, 'w', proj_name)

        # rectangular
        proj_name2 = 'rectangular_project_' + section + '.xy'
        rectangular_project.append( proj_name2 )
        rm_list.append( proj_name2 )
        lon, lat, dist = np.loadtxt( proj_name, unpack=True )
        xx = np.radians( dist ) * pid_d['radius_km']
        rr = np.tile( 0, len( xx ) ) # track at 0 km depth
        np.savetxt( proj_name2, np.column_stack( (xx, rr) ) )

    # store project files for processing and plotting routines
    func_d['annular_project'] = annular_project
    func_d['rectangular_project'] = rectangular_project

#====================================================================
#====================================================================
#====================================================================
def make_map_postscript( master_d, tt ):

    '''Make summary map.'''

    control_d = master_d['control_d']
    coor_d = master_d['coor_d']
    func_d = master_d['func_d']
    geoframe_d = master_d['geoframe_d']
    pid_d = master_d['pid_d']
    time_d = master_d['control_d']['time_d']
    datafile = pid_d['datafile']
    FULL_SPHERE = pid_d['FULL_SPHERE']
    ivel_prefix = control_d.get('ivel_prefix', None)
    nodez = pid_d['nodez']
    REGIONAL = pid_d['REGIONAL']
    rm_list = func_d['rm_list']

    # positions for all figures
    X_pos = [0.5,4.5]*4
    Y_pos = [8.375,8.375,5.75,5.75,3.125,3.125,0.5,0.5]
    # XXX DJB
    #znode_list = [53,52,51,50,49,48,47,46]
    znode_list = [64,62,60,59,58,57,56,54]

    runtime_Myr = time_d['runtime_Myr'][tt]
    age_Ma = time_d['age_Ma'][tt]
    age_int = int(round(age_Ma,0)) # to get ivel file
    time = time_d['time_list'][tt]

    if ivel_prefix:
        ivel_filename = ivel_prefix + str(age_int)
        if FULL_SPHERE: ivel_filename += '.#'
        ivel_data = Core_Citcom.read_cap_files_to_cap_list( pid_d, ivel_filename )

        control_d['coord_file'] = control_d.get('coord_dir','') + '/' + pid_d['datafile']+'.coord.#'
        control_d['OUTPUT_IVEL'] = False
        write_coordinates_by_cap( master_d )

    # postscript name
    ps = control_d['prefix'] + '_Map_%(time)s.ps' % vars()

    arg = 'PAGE_ORIENTATION portrait'
    callgmt( 'gmtset', arg )

    opts_d = Core_GMT.start_postscript( ps )

    # loop over figures
    for nn, znode in enumerate( znode_list ):

        if FULL_SPHERE:
            # XXX DJB - zoom in on Laramide flat slab
            #opts_d['B'] = 'a10/a10'
            #opts_d['J'] = 'M277/2'
            #opts_d['R'] = '250/294/25/55'

            # XXX DJB - zoom in on Izu-Bonin-Marianas
            opts_d['B'] = 'a10/a10'
            opts_d['J'] = 'M140/2'
            opts_d['R'] = '120/160/-10/30'

            # for entire globe
            #opts_d['B'] = 'a30'
            #opts_d['J'] = 'H4'
            #opts_d['R'] = 'g'
        elif REGIONAL:
            lon_min = pid_d['lon_min']
            lon_max = pid_d['lon_max']
            lat_min = pid_d['lat_min']
            lat_max = pid_d['lat_max']
            opts_d['B'] = 'a10/a5::WeSn'
            opts_d['J'] = 'X3.5/2.125'
            opts_d['R'] = '%(lon_min)s/%(lon_max)s/%(lat_min)s/%(lat_max)s' % vars()

        X = apos(X_pos[nn])
        Y = apos(Y_pos[nn])
        opts_d['X'] = X
        opts_d['Y'] = Y
        depth = int(coor_d['depth_km'][znode])

        # grdimage temperature
        # XXX DJB
        #grid = 'grid/' + datafile + '.temp.%(depth)s.%(time)s.grd' % vars()
        #opts_d['C'] = 'temp.cpt'
        #callgmt( 'grdimage', grid, opts_d, '>>', ps )
        #del opts_d['B']
        #del opts_d['C']

        # grdimage age
        # XXX DJB
        grid = '/net/beno/raid2/nflament/Agegrids/20130828_rev210/Mask/agegrid_final_mask_%(age_int)s.grd' % vars()
        opts_d['C'] = 'age.cpt'
        callgmt( 'grdimage', grid, opts_d, '>>', ps )
        del opts_d['B']
        del opts_d['C']

        # overlay GPlates line data for global models only
        if FULL_SPHERE:
            #W = '3,grey'
            #Core_GMT.plot_gplates_coastline( geoframe_d, opts_d, ps, age_int, W )
            #W = '3,yellow'
            #Core_GMT.plot_gplates_ridge_and_transform( geoframe_d, opts_d, ps, age_int, W )
            W = '3,black'
            Core_GMT.plot_gplates_slab_polygon( geoframe_d, opts_d, ps, age_int, W )
            W = '3,black'
            G = 'black'
            Core_GMT.plot_gplates_sawtooth_subduction( geoframe_d, opts_d, ps, age_int, W, G )
            W = '3,black'
            G = 'white'
            Core_GMT.plot_gplates_sawtooth_leading_edge( geoframe_d, opts_d, ps, age_int, W, G )

        # overlay psbasemap again to estimate location of cross-sections
        #opts_d['B'] = 'a10g10/a10g10'
        #callgmt( 'psbasemap', '', opts_d, '>>', ps )
        #del opts_d['B']

        if REGIONAL:
            for cc in range( pid_d['nproc_surf'] ):
                # coarse mesh that ivels are constructed using
                xyz_file = func_d['coarse_coor_cap_names'][cc]
                rm_list.append( xyz_file )
                opts_d['S'] = 'c0.03'
                callgmt( 'psxy', xyz_file, opts_d, '>>', ps )
                del opts_d['S']

        if ivel_prefix:
                # extract ivels for this depth
                for cc in range( pid_d['nproc_surf'] ):
                    ivel_slice = ivel_data[cc][znode::nodez]
                    ivel_stencil = np.array([entry[3] for entry in ivel_slice])
                    # where ivels are not applied (stencil=2, ignored)
                    xyz_filename = 'ivel.slice.2.xyz'
                    rm_list.append( xyz_filename )
                    index = np.where( ivel_stencil==2 )[0]
                    coord_for_index = np.array(func_d['coor_by_cap'][cc])[index]
                    np.savetxt( xyz_filename, coord_for_index )
                    opts_d['S'] = 'c0.03'
                    opts_d['G'] = 'white'
                    if REGIONAL:
                        callgmt( 'psxy', xyz_filename, opts_d, '>>', ps )
                    del opts_d['S']
                    del opts_d['G']
                    # where ivels are applied (stencil=1)
                    xyz_filename = 'ivel.slice.1.xyz'
                    rm_list.append( xyz_filename )
                    index = np.where( ivel_stencil==1 )[0]
                    coord_for_index = np.array(func_d['coor_by_cap'][cc])[index]
                    np.savetxt( xyz_filename, coord_for_index )
                    opts_d['S'] = 'c0.03'
                    opts_d['G'] = 'purple'
                    callgmt( 'psxy', xyz_filename, opts_d, '>>', ps ) 
                    del opts_d['S']
                    del opts_d['G']


        # plot profiles
        for section in control_d['_SECTIONS_']:

            # section dictionary
            section_d = control_d[ section ]
            arg = 'annular_project_' + section + '.xy -W5,white'
            callgmt( 'psxy', arg, opts_d, '>>', ps )
            arg = 'annular_project_' + section + '.xy -W3,black,-'
            callgmt( 'psxy', arg, opts_d, '>>', ps )

            # label start (X) and end (X') points
            lon0 = section_d['lon0']
            lat0 = section_d['lat0']
            lon1 = section_d['lon1']
            lat1 = section_d['lat1']
            letter = section.split('_')[1]
            opts_d['N'] = ' '
            opts_d['W'] = 'white'
            text = '%(lon0)s %(lat0)s 8 0 4 MC %(letter)s\n' % vars()
            text += '''%(lon1)s %(lat1)s 8 0 4 MC %(letter)s'\n''' % vars()
            text += 'EOF'
            callgmt( 'pstext', '', opts_d, '<< EOF >>', ps + '\n' + text )
            del opts_d['N']
            del opts_d['W']

        # plot depth label
        strdepth = str(depth) + ' km'
        cmd = '-R0/8.5/0/11 -Jx1.0'
        cmd += ' -X%(X)s -Y%(Y)s -K -O' % vars()
        X_text = 0.03
        Y_text = 2.25
        text = '%(X_text)s %(Y_text)s 12 0 4 ML %(strdepth)s\nEOF' % vars()
        callgmt( 'pstext', cmd, '', '<< EOF >>', ps + '\n' + text )

    # psscale
    psscaleopts_d = {}
    psscaleopts_d['B'] = 'a30'
    psscaleopts_d['C'] = 'age.cpt'
    psscaleopts_d['D'] = '5.0/0.5/1.5/0.125h'
    psscaleopts_d['K'] = ' '
    psscaleopts_d['O'] = ' '
    callgmt( 'psscale', '', psscaleopts_d, '>>', ps )

    Core_GMT.end_postscript( ps )

    return ps

#====================================================================
#====================================================================
#====================================================================
def print_integral_data( opts_d, master_X, master_Y, ps, slab_int_a,
        lith_int_a ):

    '''Report (print) integral data.'''

    intopts_d = opts_d.copy()
    intopts_d['J'] = 'x1'
    intopts_d['R'] = '0/8.5/0/11'
    intopts_d['X'] = master_X
    intopts_d['Y'] = master_Y

    ratio = slab_int_a / lith_int_a

    text = '5.5 1.2 12 0 4 ML temp\n'
    text += '6.1 1.2 12 0 4 ML velo\n'
    text += '6.8 1.2 12 0 4 ML flux\n'
    text += '5.1 1.0 12 0 4 ML slab\n'
    text += '5.5 1.0 12 0 4 ML %0.2f\n' % slab_int_a[0]
    text += '6.1 1.0 12 0 4 ML %0.2f\n' % slab_int_a[1]
    text += '6.8 1.0 12 0 4 ML %0.2f\n' % slab_int_a[2]
    text += '5.1 0.8 12 0 4 ML lith\n'
    text += '5.5 0.8 12 0 4 ML %0.2f\n' % lith_int_a[0]
    text += '6.1 0.8 12 0 4 ML %0.2f\n' % lith_int_a[1]
    text += '6.8 0.8 12 0 4 ML %0.2f\n' % lith_int_a[2]
    text += '5.1 0.6 12 0 4 ML ratio\n'
    text += '5.5 0.6 12 0 4 ML %0.2f\n' % ratio[0]
    text += '6.1 0.6 12 0 4 ML %0.2f\n' % ratio[1]
    text += '6.8 0.6 12 0 4 ML %0.2f\n' % ratio[2]
    text += 'EOF'

    callgmt( 'pstext', '', intopts_d, '<< EOF >>', ps + '\n' + text )

#====================================================================
#====================================================================
#====================================================================
def gmtset():

    '''gmtset commands.'''

    callgmt( 'gmtset', 'ANNOT_FONT_SIZE_PRIMARY', '', '', '10p' )
    callgmt( 'gmtset', 'ANNOT_FONT_PRIMARY', '', '', '4' )
    callgmt( 'gmtset', 'ANNOT_OFFSET_PRIMARY', '', '', '0.05' )
    callgmt( 'gmtset', 'HEADER_FONT', '', '', '4' )
    callgmt( 'gmtset', 'HEADER_FONT_SIZE', '', '', '12p' )
    callgmt( 'gmtset', 'HEADER_OFFSET', '', '', '-0.1' )
    callgmt( 'gmtset', 'LABEL_FONT', '', '', '4' )
    callgmt( 'gmtset', 'LABEL_FONT_SIZE', '', '', '12' )
    callgmt( 'gmtset', 'LABEL_OFFSET', '', '', '0.0' )
    callgmt( 'gmtset', 'PAGE_ORIENTATION', '', '', 'portrait' )

#====================================================================
#====================================================================
#====================================================================
def lith_buoyancy_profiles( control_d, section, x_trench ):

    '''Description.'''

    # parameters from dictionary
    lith_profile_distance = float( control_d['lith_profile_distance'] )
    res = float( control_d['profile_resolution'] ) # km

    xpos = x_trench - lith_profile_distance
    depth_min = 0 # km
    depth_max = 300.0 # km
    cc = int((depth_max-depth_min)/res)

    # store all profiles names for plotting
    out_l = []

    # *** profiles across lithosphere ***
    dist = np.tile( xpos, cc )
    depth = [depth_min+pp*res for pp in range(cc)]
    xy = 'lithp_' + section + '.c.xy'
    out_l.append( xy )
    np.savetxt( xy, np.column_stack( (dist,depth) ) )

    # get values from grids along profile line
    val_list = []
    for field in ['temp','tangent']:
        xy_out = 'lithp_' + section + '_' + field + '.c.xy'
        grid = field + '_' + section + '.c.grd'
        Core_Util.find_value_on_line( xy, grid, xy_out ) # xy defined above
        dist, val = np.loadtxt( xy_out, usecols=(1,2), unpack=True )
        val_list.append( val )
        np.savetxt( xy_out, np.column_stack( (dist,val) ) )
        out_l.append( xy_out )

    # integrals across lithosphere
    temp = val_list[0]
    velo = val_list[1]
    lithtempint = abs(round(simps(temp-1, x=None, dx=res),2))

    # velo through profile surface
    lithveloint = abs(round(simps(velo, x=None, dx=res),2))
    lithveloint /= depth_max

    # buoyancy flux through profile surface
    lithflux = -(temp-1)*velo
    xy = 'lithp_' + section + '_flux.c.xy'
    out_l.append( xy )
    np.savetxt( xy, np.column_stack( (dist, lithflux) ) )
    lithfluxint = abs(round(simps((temp-1)*velo, x=None, dx=res),2))

    int_l = np.array( [lithtempint, lithveloint, lithfluxint] )

    return out_l, int_l

#====================================================================
#====================================================================
#====================================================================
def slab_buoyancy_profiles( control_d, section, section_d, x_trench, rm_list ):

    '''Description.'''

    # parameters from dictionary
    res = float( control_d['profile_resolution'] )
    slab_profile_depth = float( control_d['slab_profile_depth'] )
    slab_dip = float( section_d['slab_dip'] )
    slab_profile_width = float( control_d['slab_profile_width'] )
    profile_dip = float( section_d['slab_dip'] ) - 90

    # store all profile names for plotting
    out_l = []

    # *** profiles across slab ***
    # center point of profile about slab (auto-center)
    centerx = x_trench
    centery = slab_profile_depth
    if slab_profile_depth < 700.0:
        centerx += slab_profile_depth / np.tan( np.radians( slab_dip ) )
    else:
        centerx += 700.0/np.tan(np.radians( slab_dip ))

    # profile line to use for tracking
    xlist = []
    ylist = []
    cc = int(slab_profile_width/res/2.0)
    for pp in range(-cc,cc+1):
        x = centerx+pp*res*np.cos(np.radians( profile_dip ))
        y = centery+pp*res*np.sin(np.radians( profile_dip ))
        xlist.append( x )
        ylist.append( y )
    xy = 'slabp_' + section + '.c.xy'
    out_l.append( xy )
    np.savetxt( xy, np.column_stack( (xlist, ylist)), fmt='%f %f' )

    # get values from grids along profile line
    val_list = []
    dist = [dd*res for dd in range(cc*2+1)]

    for field in ['temp', 'tangent', 'radial']:
        xy_out = 'slabp_' + section +'_' + field + '.c.xy'
        rm_list.append( xy_out )
        grid = field + '_' + section + '.c.grd'
        Core_Util.find_value_on_line( xy, grid, xy_out ) # xy defined above
        val = np.loadtxt( xy_out, usecols=(2,), unpack=True )
        val_list.append( val )
        np.savetxt( xy_out, np.column_stack( (dist,val) ) )
    out_l.append( 'slabp_' + section + '_temp.c.xy' )

    # integrals across slab
    temp = val_list[0]
    vyvelo = val_list[1]
    vzvelo = val_list[2]
    #ddip = np.radians( slab_dip - slab_profile_dip )
    #if slab_profile_depth < 660.0:
    #    slabtempint = abs(round(simps(temp-1, x=None, dx=res)*np.sin(ddip),2))
    #else:
    slabtempint = abs(round(simps(temp-1, x=None, dx=res),2))

    # velo through profile surface
    vycomp = vyvelo*np.sin(np.radians(abs(profile_dip)))
    vzcomp = -vzvelo*np.cos(np.radians(abs(profile_dip)))
    velo = vycomp + vzcomp
    xy = 'slabp_' + section + '_velo.c.xy'
    out_l.append( xy )
    np.savetxt( xy, np.column_stack( (dist,velo) ) )
    # average velocity along profile (per unit km)
    slabveloint = abs(round(simps(velo, x=None, dx=res),2))
    slabveloint /= slab_profile_width

    # buoyancy flux through profile surface
    slabflux = -(temp-1)*velo
    xy = 'slabp_' + section + '_flux.c.xy'
    out_l.append( xy )
    np.savetxt( xy, np.column_stack( (dist,slabflux) ) )
    slabfluxint = abs(round(simps((temp-1)*velo, x=None, dx=res),2))

    int_l = np.array( [slabtempint, slabveloint, slabfluxint] )

    return out_l, int_l, rm_list

#====================================================================
#====================================================================
#====================================================================
def get_bm_and_grid_file_names( xyz_file ):

    '''Description.'''

    # block = blockmedian
    block_name = xyz_file.rstrip('xyz') + 'bm.xyz'
    grid_name = xyz_file.rstrip('xyz') + 'grd'

    return block_name, grid_name

#====================================================================
#====================================================================
#====================================================================
def make_cpts( master_d ):

    '''Make cpt files.'''

    rm_list = master_d['func_d']['rm_list']

    cpt_name = 'age.cpt'
    rm_list.append( cpt_name )
    cptopts_d = {}
    cptopts_d['C'] = 'rainbow'
    cptopts_d['I'] = ' '
    cptopts_d['T'] = '0/120/5'
    callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    cpt_name = 'temp.cpt'
    rm_list.append( cpt_name )
    cptopts_d = {}
    cptopts_d['C'] = 'jet'
    cptopts_d['D'] = ' '
    cptopts_d['T'] = '0/1/0.1'
    # XXX DJB
    #cptopts_d['Z'] = ' ' 
    callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    cpt_name = 'visc.cpt'
    rm_list.append( cpt_name )
    cptopts_d = {}
    cptopts_d['C'] = 'jet'
    cptopts_d['I'] = ' '
    cptopts_d['T'] = '-1/3/0.25'
    cptopts_d['Z'] = ' '
    callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    cpt_name = 'divv_norm.cpt'
    rm_list.append( cpt_name )
    cptopts_d = {}
    cptopts_d['C'] = 'jet'
    cptopts_d['T'] = '-3/0/0.5'
    cptopts_d['Z'] = ' '
    callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    #cpt_name = 'pressure.cpt'
    #rm_list.append( cpt_name )
    #cptopts_d = {}
    #cptopts_d['C'] = 'jet'
    #cptopts_d['T'] = '-5E5/5E5/1E3'
    #cptopts_d['Z'] = ' ' 
    #callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    cpt_name = 'velo.cpt'
    rm_list.append( cpt_name )
    cptopts_d = {}
    cptopts_d['C'] = 'jet'
    cptopts_d['D'] = ' '
    #cptopts_d['T'] = '-7.5/7.5/1.5'
    cptopts_d['T'] = '-5/5/0.5' #'-6/12/1'
    #cptopts_d['Z'] = ' ' 
    callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    cpt_name = 'vmag.cpt'
    rm_list.append( cpt_name )
    cptopts_d = {}
    cptopts_d['C'] = 'jet'
    cptopts_d['D'] = ' '
    cptopts_d['T'] = '0/5/0.25'
    #cptopts_d['Z'] = ' '
    callgmt( 'makecpt', '', cptopts_d, '>', cpt_name )

    subprocess.call( 'cp velo.cpt tangent.cpt', shell=True )
    subprocess.call( 'cp velo.cpt radial.cpt', shell=True )
    #subprocess.call( 'cp ' + cpt_name + ' vmag.cpt', shell=True )
    rm_list.extend( ['tangent.cpt', 'radial.cpt', 'vmag.cpt'] )

    cpt_name = 'sten_temp.cont'
    rm_list.append( cpt_name )
    cfile = open( cpt_name, 'w' )
    cfile.write( '0.1 C\n' )
    cfile.write( '0.5 C\n' )
    cfile.write( '0.9 C\n' )
    cfile.close()

    cpt_name = 'temp.cont'
    rm_list.append( cpt_name )
    cfile = open( cpt_name, 'w' )
    for ii in range(5):
        val = 0.1+ii*0.2
        cfile.write( '%(val)s C\n' % vars() )
    cfile.close()

    cpt_name = 'agetemp.cont'
    rm_list.append( cpt_name )
    cfile = open( cpt_name, 'w' )
    cfile.write( '0.25 C\n0.5 C\n0.75 C' )
    cfile.close()

    return rm_list

#====================================================================
#====================================================================
#====================================================================
def make_example_config_file():

    text = '''prefix = ram01
pid_file = pid19335.cfg
time_spec = 0/150/10

profile_resolution = 1 ; resolution (km)
lith_profile_distance = 100 ; from subduction zone (km)
slab_profile_width = 400 ; width of slab profile (km)
slab_profile_depth = 350 ; depth of slab profile (km)

# thermal age grids produced by Create_History.py
lith_thermal_age_grids = /net/beno2/nobackup1/danb/input/mkhist/regional/ramhist/age_grid/lith_age_

# plot ivels on map summary (do not plot if commented out)
ivel_prefix = /net/beno2/nobackup1/danb/datalib-local/GMWDIR2013.2_rev218/ivel/ivel.dat

# for regional (often)
velocity_scale = 12

# for global (often)
#velocity_scale = 20

# plot cross-section summary
PLOT_CROSS_SECTIONS = True

# plot map summary
PLOT_MAPS = True


# some regional profiles

[Figure_A]
lon0 = 0.0 ; start lon
lat0 = 0.0 ; start lat
lon1 = 57.2957795 ; end lon
lat1 = 0.0; end lat
slab_dip = 45 ; dip of slab (degrees)

[Figure_B]
lon0 = 40.0
lat0 = 0.0
lon1 = 54.0
lat1 = 14.0
slab_dip = 45

[Figure_C]
lon0 = 54.0
lat0 = 0.0
lon1 = 40.0
lat1 = 14.0
slab_dip = 45

# some global profiles

#[loc_A]
## Izu Bonin Mariana Trench
## no dip set, so defaults to 45 degrees
#slab_dip = 45
#lon0 = 150
#lat0 = 20
#lon1 = 130
#lat1 = 0

#[loc_B]
## Aleutian and Bering Sea
## no dip set, so defaults to 45 degrees
#slab_dip = 45
#lon0 = 170
#lat0 = 40
#lon1 = 180
#lat1 = 70

#[loc_C]
## JF/NA
#slab_dip = 20
#lon0 = 200
#lat0 = 30
#lon1 = 270
#lat1 = 50

#[loc_D]
## South America
## largely no dip set, so defaults to 45 degrees
#slab_dip = 45
#lon0 = 260
#lat0 = -10
#lon1 = 310
#lat1 = -10

#[loc_E]
## SW Pacific
#slab_dip = 45
#lon0 = 180
#lat0 = -20
#lon1 = 160
#lat1 = -30
''' % vars()

    print( text )

#====================================================================
#====================================================================
#====================================================================

if __name__ == "__main__":

    # check for script called wih no arguments
    if len(sys.argv) != 2:
        usage()
        sys.exit(-1)

    # create example config file 
    if sys.argv[1] == '-e':
        make_example_config_file()
        sys.exit(0)

    # run the main script workflow
    main()
    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
