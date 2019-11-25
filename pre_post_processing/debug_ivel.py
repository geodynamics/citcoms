#!/usr/bin/env python

import Core_Citcom, Core_GMT, Core_Util, subprocess
import numpy as np
from Core_GMT import callgmt

# standard arguments
geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
opts2_d = {'R':'g','J':'H180/8','X':'a1.5','Y':'a1.5'}
str_list = ['vx','vy','vz']

age_list = [29]

cmd = 'LABEL_FONT_SIZE 14p'
cmd += ' LABEL_OFFSET 0.05'
callgmt( 'gmtset', cmd )

for age in age_list:
    print( 'age=', age )
    filename = 'debug_ivel.%(age)s.xy' % vars()
    lon, lat, subparallel, sub, vx, vy, vz = np.loadtxt( filename, unpack=True )

    for nn, comp in enumerate([vx,vy,vz]):

        str_comp = str_list[nn]
        temp_name = 'output.%(age)s.xyz' % vars()
        np.savetxt( temp_name, np.column_stack( (lon,lat,comp) ) )
        ps = 'output.%(age)s.%(str_comp)s.ps' % vars()

        opts_d = Core_GMT.start_postscript( ps )
        opts_d.update( opts2_d )

        grid_name = geoframe_d['gplates_velo_grid_dir']
        grid_name += '/gplates_%(str_comp)s.0.%(age)s.grd' % vars()
        cmd = grid_name + ' -Cvelocity.cpt'
        callgmt( 'grdimage', cmd, opts_d, '>>', ps )

        W = '3,red'
        Core_GMT.plot_gplates_ridge_and_transform( geoframe_d, opts_d, ps, age, W )

        G = 'black'
        W = '3,black'
        Core_GMT.plot_gplates_sawtooth_subduction( geoframe_d, opts_d, ps, age, W, G )

        cmd = ''' -Ba30g10/a30g10:."%(str_comp)s:"''' % vars()
        callgmt( 'psbasemap', cmd, opts_d, '>>', ps )

        cmd = temp_name + ' -m -Sc0.05 -Cvelocity.cpt' % vars()
        callgmt( 'psxy', cmd, opts_d, '>>', ps )

        Core_GMT.end_postscript( ps )

    # magnitude of velocity
    vmag = np.sqrt( np.square(vx) + np.square(vy) + np.square(vz) )
    #vmag = np.sqrt( np.square(subparallel) + np.square(sub) )
    np.savetxt( temp_name, np.column_stack( (lon,lat,vmag) ) )

    ps = 'output.%(age)s.vmag.ps' % vars()
    opts_d = Core_GMT.start_postscript( ps )
    opts_d.update( opts2_d )

    grid_name = geoframe_d['gplates_velo_grid_dir']
    grid_name += '/gplates_vmag.0.%(age)s.grd' % vars()
    cmd = grid_name + ' -Cvelocity_mag.cpt'
    callgmt( 'grdimage', cmd, opts_d, '>>', ps )

    W = '3,red'
    Core_GMT.plot_gplates_ridge_and_transform( geoframe_d, opts_d, ps, age, W )

    G = 'black'
    W = '3,black'
    Core_GMT.plot_gplates_sawtooth_subduction( geoframe_d, opts_d, ps, age, W, G )

    cmd = ''' -Ba30g10/a30g10:."vmag":''' % vars()
    callgmt( 'psbasemap', cmd, opts_d, '>>', ps )

    cmd = temp_name + ' -m -Sc0.05 -Cvelocity_mag.cpt' % vars()
    callgmt( 'psxy', cmd, opts_d, '>>', ps )

    del opts_d['J']
    del opts_d['R']
    cmd = ' -Cvelocity_mag.cpt -D4/-0.25/3/0.15h -Ba5f1:"velocity (cm/yr)":'
    callgmt( 'psscale', cmd, opts_d, '>>', ps )
    Core_GMT.end_postscript( ps )
