#!/usr/bin/env python
#
#=====================================================================
#
#               Python Scripts for CitcomS Data Assimilation
#                  ---------------------------------
#
#                              Authors:
#                    Dan J. Bower, Nicolas Flament
#          (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#
#
#=====================================================================
#
#  Copyright 2006-2015, by the California Institute of Technology.
#
#  Last update: 29th January 2015 by DJB
#=====================================================================

import Core_GMT5, Core_Util, sys
from Core_GMT5 import callgmt
import subprocess
from subprocess import Popen, PIPE
from Core_Util import now
import scipy.optimize
import numpy as np

verbose = True

#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    '''print usage message and exit'''

    print('''process_topography_grid.py pid_file surf_topo_grid WATER_LOAD

where:

    pid_file      :   CitcomS pid_file

    surf_topo_grid:   Surface topography GMT grid (non-dimensional)
                      created using grid_maker.py or another script.
                      This grid contains the raw normal (radial)
                      stress output from CitcomS.  Do not
                      dimensionalise before running this script!

    WATER_LOAD    :   Flag (1 on, 0 off) to also build water-loaded
                      topography grid

''')

    sys.exit(0)

#=====================================================================
#=====================================================================
#=====================================================================
def main():

    '''Main sequence of script actions.'''

    # input arguments
    pid_file = sys.argv[1]
    grid = sys.argv[2]
    WATER_LOAD = sys.argv[3]

    pid_d = Core_Util.parse_configuration_file( pid_file )

    # XXX HARD-CODED
    pid_d['rho_mantle'] = 3300 # kg m^-3
    pid_d['rho_water'] = 1025 # kg m^-3

    # scaled (dimensional) air loaded topography grid is required
    # for water loading
    air_grid = air_loaded_topography( pid_d, grid )

    if WATER_LOAD:
        water_loaded_topography( pid_d, air_grid )

#=====================================================================
#=====================================================================
#=====================================================================
def water_loaded_topography( pid_d, air_grid ):

    '''build water_loaded topography grid.'''

    # grdinfo 
    zmin, zmax = grdinfo( air_grid )

    # volume of the oceans based on etopo1
    # In Flament et al. (2008) I had 1.36e18+/-2e17 m^3
    volume_true = 1.33702586599e+18 # units of m^3

    # loading factor is a constant that quantifies the ratio of
    # water-loaded topography to air-loaded topography for a given
    # (radial) normal stress
    # derived from dh_air = dP / (drho_a * g)
    # where drho_a = 3300 - 0
    # dh_water = dP / (drho_w * g)
    # where drho_w = 3300 - 1025 (1025 approx density of seawater)
    # in Flament et al. (2014) I used rho_m=3340 kg m^-3 and rho_water=1030 kg m^-3 
    ## so that Loading_fact=3340/(3340-1030)
    loading_factor = pid_d['rho_mantle']
    loading_factor /=  pid_d['rho_mantle'] - pid_d['rho_water']
    if verbose: print( now(), 'loading_factor=', loading_factor )

    # volume of the oceans with water removed
    # all calculations are then computed in the 'air-loaded' regime
    volume_target = volume_true/loading_factor

    # find sea-level (contour) using van Wijngaarden-Deker-Brent method
    # see scipy documentation online
    # it's a method for finding the root of a function in a given interval
    contour = scipy.optimize.brentq( volume_error, zmin+1,
        zmax, args=( zmin, air_grid, volume_target))

    if verbose:
        print( now(), 'found solution:')
        print( now(), 'contour=', contour )
        volume = volume_of_basin( contour, zmin, air_grid )
        print( now(), 'volume=', volume )
        print( now(), 'volume_target=', volume_target )
        rel_err = np.abs( (volume_target-volume)/volume_target)
        print( now(), 'relative_err=', rel_err )

    # remove list for clean up
    remove_l = []

    # subtract contour from original, air-loaded topography
    prefix = air_grid.rstrip('al-dim.nc')
    prefix += '-wl-dim'
    output_grd1 = prefix + '1.nc'
    remove_l.append( output_grd1 )
    arg = air_grid + ' ' + str(contour) + ' SUB' 
    callgmt( 'grdmath', arg, '', '=', output_grd1 )

    # remove positive values (that should be air-loaded) from that grid
    output_grd2 = prefix + '2.nc'
    remove_l.append( output_grd2 )
    cmd = output_grd1 + ' -Sa0/NaN'
    callgmt( 'grdclip', cmd, '', '', '-G' + output_grd2 )

    # water loading of the negative values (oceans)
    output_grd3 = prefix + '3.nc'
    remove_l.append( output_grd3 )
    arg = output_grd2 + ' ' + str(loading_factor) + ' MUL'
    callgmt( 'grdmath', arg, '', '=', output_grd3 )

    # stitching air-loaded continents and water-loaded oceans together
    output_grd4 = prefix + '.nc'
    arg = output_grd3 + ' ' + output_grd1 + ' AND'
    callgmt( 'grdmath', arg, '', '=', output_grd4 )

    Core_Util.remove_files( remove_l )

    print( 'Done!' )

#=====================================================================
#=====================================================================
#=====================================================================
def air_loaded_topography( pid_d, grid ):

    '''build air-loaded topography grids:
        (1) ...-al-dim.nc:         dimensional
        (2) ...-al-mean-dim.nc:    mean (dimensional) (i.e., constant grid)
        (3) ...-al-no-mean-dim.nc: dimensional with mean removed'''

    drho = pid_d['rho_mantle']
    g = pid_d['gravacc']
    R0 = pid_d['radius']
    eta = pid_d['refvisc']
    kappa = pid_d['thermdiff']

    scale = 1/(drho*g) * (eta*kappa)/(R0*R0)

    prefix = grid.rstrip('.grd')

    output_grd = prefix + '-al-dim.nc'
    arg = grid + ' ' + str(scale) + ' MUL' 
    callgmt( 'grdmath', arg, '', '=', output_grd )

    mean_nc = prefix + '-al-mean-dim.nc'
    arg = output_grd + ' MEAN'
    callgmt( 'grdmath', arg, '', '=', mean_nc )   

    output_grd2 = prefix + '-al-no-mean-dim.nc'
    arg = output_grd + ' ' + mean_nc + ' SUB'
    callgmt( 'grdmath', arg, '', '=', output_grd2 )

    return output_grd

#=====================================================================
#=====================================================================
#=====================================================================
def grdinfo( grid ):

    '''return zmin and zmax of a GMT grd file'''

    cmd = ['gmt', 'grdinfo', grid ]
    p1 = Popen( cmd, stdout=PIPE, stderr=PIPE )
    stdout, stderr = p1.communicate()
    stdout = stdout.decode()

    zmin = float(stdout.split('z_min:')[1].split()[0])
    zmax = float(stdout.split('z_max:')[1].split()[0])

    if verbose:
        print( now(), 'zmin=', zmin, ', zmax=', zmax )

    return (zmin, zmax)

#=====================================================================
#=====================================================================
#=====================================================================
def volume_of_basin( high, low, grid ):

    '''compute volume of a basin between a low and a high contour.
       See GMT5 man pages for information regarding the Cr option.'''

    arg = '-Cr%(low)s/%(high)s' % vars()
    cmd = [ 'gmt', 'grdvolume', grid, arg, '-Se' ]
    p1 = Popen( cmd, stdout=PIPE, stderr=PIPE )

    stdout, stderr = p1.communicate()
    stdout = stdout.decode()
    volume = float(stdout.split()[2])

    return volume

#=====================================================================
#=====================================================================
#=====================================================================
def volume_error( high, low, grid, volume_target ):

    '''compute volume error of a basin relative to a target volume.'''

    volume = volume_of_basin( high, low, grid )
    err = volume - volume_target

    if verbose:
        print( now(), 'contour=', high )
        print( now(), 'volume=', volume )
        print( now(), 'volume_target=', volume_target )
        print( now(), 'error=', err )

    return err

#=====================================================================
#=====================================================================
#=====================================================================

if __name__ == "__main__":

    if len(sys.argv) < 4:
        usage()
        sys.exit(1)
    else:
        main()
