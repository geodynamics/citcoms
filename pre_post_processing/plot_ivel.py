#!/usr/bin/env python

import Core_Citcom, Core_Util, subprocess
import numpy as np
from Core_GMT import callgmt
import Core_GMT

# input parameters
age = 29
levels = 3

# parameters
master_d = Core_Citcom.get_all_pid_data( 'pid23039.cfg' )
coor_d = master_d['coor_d']
geoframe_d = master_d['geoframe_d']
pid_d = master_d['pid_d']
nodez = pid_d['nodez']
nproc_surf = pid_d['nproc_surf']

# location of files (file template)
coord_template = 'coord/ghk90.coord.#'
ivel_template = 'ivel/levels_%(levels)s/ivel.dat%(age)d.#' % vars()

# by cap
coord_by_cap = Core_Citcom.read_proc_files_to_cap_list( pid_d, coord_template, 'coord' )
ivel_by_cap = Core_Citcom.read_cap_files_to_cap_list( pid_d, ivel_template )

# bounds for caps
bounds = []
bounds.append('0/89.8757067303/-0.124313117575/89.8242187257')
bounds.append('45.0001338775/134.999857322/-35.2638146929/35.2638120338')
bounds.append('90.1243131176/180.000019848/-89.8242363963/0.124293269732')
bounds.append('64.3885959463/180.000019848/0.124293269732/89.8242187257')
bounds.append('134.999857322/225.000125077/-35.0882031287/35.4397788319')
bounds.append('180.000019848/269.875726578/-89.8242363963/0.124293269732')
bounds.append('180.000019848/295.611443749/0.124293269732/89.8242187257')
bounds.append('225.000125077/314.99987717/-35.2638146929/35.2638120338')
bounds.append('244.388558498/359.9999824/-89.8242363963/-0.124313117575')
bounds.append('270.12427567/359.9999824/-0.124313117575/89.8242187257')
bounds.append('0.387219030007/359.9999824/-35.4397700318/35.0882176584')
bounds.append('0/115.611423901/-89.8242363963/-0.124313117575')

# loop over depths
for level in range( nodez-20, nodez ):

    depth_km = int(coor_d['depth_km'][level])
    print( 'working on', depth_km )

    # loop over caps
    for cc in range( nproc_surf ):

        print( 'working on cap %d' % cc )
        ivel_at_depth = ivel_by_cap[cc][level::nodez]
        coor_at_depth = coord_by_cap[cc][level::nodez]

        stencil_list = [row[3] for row in ivel_at_depth]
        stencil_array = np.array( stencil_list, dtype=int )

        # find stencil values = 1 (i.e. node turned on for ivel)
        itemindex = np.array(np.where( stencil_array == 1 ))[0]
        coord_extract = np.array(coor_at_depth)[itemindex]

        # gmt commands
        ps = 'ivel/levels_%(levels)s/ivel.%(depth_km)skm.%(cc)s.ps' % vars()
        arg = 'PAGE_ORIENTATION portrait'
        callgmt( 'gmtset', arg, '', '', '' )

        opts_d = Core_GMT.start_postscript( ps )

        # psbasemap
        lon_mid = float(bounds[cc].split('/')[0])
        lon_mid += float(bounds[cc].split('/')[1])
        lon_mid *= 0.5

        opts_d['J'] = 'H%(lon_mid)s/7.5' % vars()
        opts_d['R'] = bounds[cc]
        opts_d['X'] = 'a0.5'
        opts_d['Y'] = 'a0.5'
        callgmt( 'psbasemap', '-Ba10', opts_d, '>>', ps )

        # coastlines
        W = '2,grey'
        Core_GMT.plot_gplates_coastline( geoframe_d, opts_d, ps, age, W )

        # subduction
        G = 'black'
        W = '2,black'
        Core_GMT.plot_gplates_sawtooth_subduction( geoframe_d, opts_d, ps, age, W, G )

        # plot fine mesh
        # these are output by make_history_for_age.py with DEBUG on
        # XXX DJB - commented out
        fine_filename = 'ivel/levels_%(levels)s/coord/coord.cap.%(cc)s' % vars()
        cmd = fine_filename + ' -Sc0.01 -Gblack'
        #callgmt( 'psxy', cmd, opts_d, '>>', ps )

        # plot coarse mesh
        # these are output by make_history_for_age.py with DEBUG on
        coarse_filename = 'ivel/levels_%(levels)s/coord/coarse.coord.cap.%(cc)s' % vars()
        cmd = coarse_filename + ' -Sc0.02 -Ggreen'
        callgmt( 'psxy', cmd, opts_d, '>>', ps )

        # plot nodes that are turned on for internal velocity bcs
        if coord_extract.any():
            filename = 'ivel/levels_%(levels)s/ivel.on.%(depth_km)skm.xy' % vars()
            sfile = open( filename, 'w' )
            for line in coord_extract:
                lon = np.degrees(line[1])
                lat = 90-np.degrees(line[0])
                lineout = '%(lon)s %(lat)s\n' % vars()
                sfile.write( lineout )
            sfile.close()
            cmd = filename + ' -Sc0.02 -Gred'
            callgmt( 'psxy', cmd, opts_d, '>>', ps )

        Core_GMT.end_postscript( ps )
