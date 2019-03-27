#!/usr/bin/env python
#
#=====================================================================
#
#               Python Scripts for CitcomS Data Assimilation
#                  ---------------------------------
#
#                              Authors:
#                            Dan J. Bower
#          (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#
#
#=====================================================================
#
#  Copyright 2006-2013, by the California Institute of Technology.
#
#  Last update: 13th February 2014 by DJB
#=====================================================================

import Core_Util, Core_Citcom, Core_GMT
from Core_GMT import callgmt
import numpy as np
from scipy.special import erf
from Core_Util import now
import sys

#time_spec = 520#300#150

#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    '''print usage message and exit'''

    print('''plot_lith_age.py pid_file timestep

where:

    pid_file:        CitcomS pid file

    timestep:        Timestep to process

''')

    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
def main():

    '''Main sequence of script actions.'''

    print( now(), 'plot_lith_age.py:' )
    print( now(), 'main:' )

    if len(sys.argv) != 3:
        usage()

    # parameters
    pid_file = sys.argv[1]
    time_spec = sys.argv[2]

    ### parse and setup dictionaries ###

    master_d = Core_Citcom.get_all_pid_data( pid_file )
    pid_d = master_d['pid_d']
    time_d = Core_Citcom.get_time_spec_dictionary( time_spec, master_d['time_d'])
    runtime_Myr = time_d['runtime_Myr'][0]
    age = int(round(time_d['age_Ma'][0],0))
    datafile = pid_d['datafile']
    lith_age_depth = pid_d['lith_age_depth']
    start_age = pid_d['start_age']
    time = time_d['time_list'][0]
    geoframe_d = master_d['geoframe_d']
    depth_km = master_d['coor_d']['depth_km']
    mantle_temp = pid_d['mantle_temp']
    radius = master_d['coor_d']['radius']
    radius_outer = pid_d['radius_outer']
    radius_km = pid_d['radius_km']
    scalet = pid_d['scalet']
    rm_list = [] # list of files to remove


    ###################################
    ### input directories and files ###
    ###################################

    # reconstructed cross-section (plate frame of reference)
    cross_section_dir = 'aus_xsect/'
    cross_section_name = cross_section_dir + 'reconstructed_%(age)s.00Ma.xy' % vars()

    # continental grids
    cont_dir = geoframe_d['age_grid_cont_dir'] + '/'
    cont_name = cont_dir + geoframe_d['age_grid_cont_prefix'] + '%(age)s.grd' % vars()

    # directory of lith_age3_%(age)s.grd files from make_history_for_age.py
    lith_age_dir = '/net/beno2/nobackup1/danb/global/lith_age/'
    lith_age_name = lith_age_dir + 'lith_age3_%(age)s.grd' % vars()

    ### end input directories and files ###

    ### process cross_section_name ###

    infile = open( cross_section_name, 'r' )
    lines = infile.readlines()
    infile.close
    out = []
    for line in lines:
        if line.startswith('>'):
            pass
        else:
            out.append( line.strip() )

    # profile start location
    lon0 = float( out[0].split()[0] )
    lat0 = float( out[0].split()[1] )
    print( now(), '(lon0, lat0)', lon0, lat0 )
    # profile end location
    lon1 = float( out[1].split()[0] )
    lat1 = float( out[1].split()[1] )
    print( now(), '(lon1, lat1)', lon1, lat1 )

    # min and max bounds for GMT region (R)
    lon_min = min(lon0,lon1) - 10
    lon_max = max(lon0,lon1) + 10
    lat_min = min(lat0,lat1) - 15
    lat_max = max(lat0,lat1) + 15
    print( now(), '(lon_min, lat_min)', lon_min, lat_min )
    print( now(), '(lon_max, lat_max)', lon_max, lat_max )

    # Nico's 1-D profile
    # interpolate for data points between end values
    proj_name = cross_section_name.rstrip('xy') + 'p.xy'
    rm_list.append( proj_name )
    dlon = lon1-lon0
    dlat = lat1-lat0

    outfile = open( proj_name, 'w')
    outfile.write( '%(lon0)s %(lat0)s %(lon0)s\n' % vars() )

    lon = lon0
    lat = lat0
    while 1:
        lon += dlon/500
        lat += dlat/500
        if lon <= lon1: #and lat <= lat1:
            lineout = '%(lon)s %(lat)s %(lon)s\n' % vars()
            outfile.write( lineout )
        else:
            break

    outfile.close()

    # purple circles
    # map
    lon_markers = cross_section_dir + 'lon_markers_map.xy'
    rm_list.append( lon_markers )
    ofile = open( lon_markers, 'w' )
    lon_floor = int(np.floor( lon0 ))
    lon_ceil = int(np.ceil( lon1 ))
    for lon in range( lon_floor, lon_ceil+1 ):
        if not lon % 5:
            olat = (lon-lon0)/dlon*dlat+lat0
            outline = '%(lon)s %(olat)s\n' % vars()
            ofile.write( outline )
    ofile.close()

    # annulus
    lon_markers_ann = cross_section_dir + 'lon_markers_ann.xy'
    rm_list.append( lon_markers_ann )
    plon, plat = np.loadtxt( lon_markers, unpack = True )
    prad = np.tile( radius_outer, len(plon) )
    np.savetxt( lon_markers_ann, np.column_stack( (plon,prad) ) )

    ### end process cross_section_name ###


    ### build list of temperature grids to track through ###
    # these grids must have previously been created using grid_maker.py
    gpfx = 'grid/' + datafile
    temp_list = []
    for depth in depth_km:
        gsfx = '.temp.' + str(int(depth)) + '.' + str(time) + '.grd'
        temp_list.append( gpfx + gsfx )

    # take just from 500 km depth and less
    depth_km_array = np.array( depth_km )
    znode = np.min(np.where( depth_km_array < 500 ))-1

    temp_list = temp_list[znode:]

    ### end of build temperature grid list ###


    #### idealized thermal structure from age grids
    ideal_lith_xyz = cross_section_dir + 'ideal.lith.%(age)s.xyz' % vars()
    rm_list.append( ideal_lith_xyz )
    Core_Util.find_value_on_line( proj_name, lith_age_name, ideal_lith_xyz )
    lithlon, lithlat, lithdist1, lithage_Ma = np.loadtxt( ideal_lith_xyz, unpack=True )
    lithdist = np.tile( lithdist1, pid_d['nodez'] )
    lithage_Ma = np.tile( lithage_Ma, pid_d['nodez'] )
    lithrad = [] 
    for rad in radius:
        lithrad.extend( [rad for xx in range( len(lithdist1) )] ) 
    lithrad = np.array( lithrad )
    lithtemp = erf((1.0-lithrad)/(2.0*np.sqrt(lithage_Ma/scalet)))
    lithtemp *= float( mantle_temp )


    nan = np.where(np.isnan( lithtemp ))
    #nnan = np.where(~np.isnan( lithtemp ))
    np.savetxt( ideal_lith_xyz, np.column_stack( (lithdist, lithrad, lithtemp) ) )

    #nan_values = np.ones( np.size( nan ) )*-1
    #f_handle = open( ideal_lith_xyz, 'ab')
    #np.savetxt(f_handle, np.column_stack( (lithdist[nan], lithrad[nan], nan_values) ))
    #f_handle.close()

    #### end of idealized thermal structure from age grids

    # make temperature xyz
    temp_xyz = cross_section_dir  + 'citcom.temp.%(age)s.xyz' % vars()
    rm_list.append( temp_xyz )
    # this is hacky, but loop over only the top 500 km
    master_d['coor_d']['radius'] = master_d['coor_d']['radius'][znode:]
    pao, x_ann_max = Core_Util.make_annulus_xyz( master_d, proj_name, temp_xyz, temp_list )


    ### make idealized lithosphere and citcom temperature grid ###
    blockmedian_I = '0.2/0.0035'
    surface_I = '0.1/0.00125'
    surface_T = '0.25'
    rad_in = '0.92151939'
    rad_out = '1.0'
    # for plotting data
    R_ann = str( lon0 ) + '/' + str( lon1 ) + '/' + rad_in + '/' + rad_out
    # for dimensional psbasemap
    psbase_R = str( lon0 ) + '/' + str( lon1 ) + '/' + str(5871) + '/' + str(radius_km)

    grid_names = []
    for xyz in [temp_xyz, ideal_lith_xyz]:
        block_name = xyz.rstrip('xyz') + 'b.xyz'
        rm_list.append( block_name )
        grid_name = block_name.rstrip('xyz') + 'grd'
        grid_names.append( grid_name )
        rm_list.append( grid_name )
        cmd = xyz + ' -I' + blockmedian_I + ' -R' + R_ann
        callgmt( 'blockmedian', cmd, '', '>', block_name )
        cmd = block_name + ' -I' + surface_I + ' -R' + R_ann
        cmd += ' -T' + surface_T
        cmd += ' -Ll0 -Lu1'
        callgmt( 'surface', cmd, '', '', '-G' + grid_name )

    ### end of make temperature grids ###

    ### percentage error between temperature fields ###
    cmd = grid_names[0] + ' ' + grid_names[1] + ' SUB '
    cmd += grid_names[1] + ' DIV'
    cmd += ' 100 MUL'
    temp_diff_grid = cross_section_dir + 'temp.difference.grd'
    grid_names.append( temp_diff_grid )
    rm_list.append( temp_diff_grid )
    callgmt( 'grdmath', cmd, '', '=', temp_diff_grid )

    ### end percentage error

    ### lith_age_depth overlay line
    xy = cross_section_dir + 'lith_depth.xy'
    rm_list.append( xy )
    lith_age_radius = pid_d['radius_outer']-pid_d['lith_age_depth']
    lith_depth = np.tile( lith_age_radius, len( lithdist1 ) )
    np.savetxt( xy, np.column_stack( (lithdist1, lith_depth) ) )

    ### end overlay line


    ### make cpts ###

    # age grid
    cpt_pfx = cross_section_dir
    cpt_name = cpt_pfx + 'age.cpt'
    rm_list.append( cpt_name )
    cmd = '-Crainbow -T0/370/10'
    callgmt( 'makecpt', cmd, '', '>', cpt_name )

    # continental types
    cpt_name = cpt_pfx + 'cont.cpt'
    rm_list.append( cpt_name )
    cmd = '-Crainbow -T-4/0/1'
    callgmt( 'makecpt', cmd, '', '>', cpt_name )

    # differential temperature
    cpt_name = cpt_pfx + 'diff.cpt'
    rm_list.append( cpt_name )
    cmd = '-Cpolar -T-10/10/1'
    callgmt( 'makecpt', cmd, '', '>', cpt_name )

    # temperature
    cpt_name = cpt_pfx + 'temp.cpt'
    cmd = '-Cpolar -T0/1/0.0675'
    rm_list.append( cpt_name )
    callgmt( 'makecpt', cmd, '', '>', cpt_name )

    # for temperature contours
    cpt_name = cpt_pfx + 'temp.cont'
    cmd = '-Cjet -T0.1/0.4/0.1'
    rm_list.append( cpt_name )
    callgmt( 'makecpt', cmd, '', '>', cpt_name )

    ### plotting ###
    ps = datafile + '.lith.age.analysis.%(age)sMa.ps' % vars()

    callgmt( 'gmtset', 'PAGE_ORIENTATION', '', '', 'portrait' )
    callgmt( 'gmtset', 'LABEL_FONT_SIZE', '', '', '12' )
    callgmt( 'gmtset', 'LABEL_FONT', '', '', '4' )
    callgmt( 'gmtset', 'LABEL_OFFSET', '', '', '0.02' )
    callgmt( 'gmtset', 'ANNOT_FONT_SIZE_PRIMARY', '', '', '10p' )
    callgmt( 'gmtset', 'ANNOT_FONT_PRIMARY', '', '', '4' )

    opts_d = Core_GMT.start_postscript( ps )

    # pre-initialize for pstext commands
    pstext_d = opts_d.copy()
    pstext_d['R'] = '0/8.5/0/11'
    pstext_d['J'] = 'x1.0'

    # title information
    stdin = '1 10.5 14 0 4 ML Model = %(datafile)s\n' % vars()
    stdin += '1 10.3 14 0 4 ML lith_age_depth = %(lith_age_depth)s\n' % vars()
    stdin += '7.5 10.5 14 0 4 MR Current Age = %(age)s Ma\n' % vars()
    stdin += '7.5 10.3 14 0 4 MR start_age = %(start_age)s Ma\nEOF' % vars()
    callgmt( 'pstext', '', pstext_d, '<< EOF >>', ps + '\n' + stdin )


    # plot maps #
    map_d = opts_d.copy()
    map_d['B'] = 'a20f10/a10f5::WeSn'
    map_d['R'] = '%(lon_min)s/%(lon_max)s/%(lat_min)s/%(lat_max)s' % vars()
    map_d['C'] = cross_section_dir + 'age.cpt'
    map_d['J'] = 'M3'
    map_d['X'] = 'a1'
    map_d['Y'] = 'a8'
    map_grid = lith_age_name

    callgmt( 'grdimage', lith_age_name, map_d, '>>', ps )

    C = cross_section_dir + 'age.cpt'
    cmd = '-Ba50f10:"Age (Ma)": -D2.5/7.5/2.5/0.1h -C%(C)s -K -O' % vars()
    callgmt( 'psscale', cmd, '', '>>', ps )

    del map_d['B']
    del map_d['C']
    map_d['m'] = ' '
    map_d['W'] = '5,white'
    callgmt( 'psxy', proj_name, map_d, '>>', ps )
    del map_d['m']
    del map_d['W']
    map_d['G'] = 'purple'
    map_d['S'] = 'c0.05'
    callgmt( 'psxy', lon_markers, map_d, '>>', ps )
    del map_d['G']
    del map_d['S']


    # continental types
    map_d['B'] = 'a20f10/a10f5::wESn'
    map_d['C'] = cross_section_dir + 'cont.cpt'
    map_d['X'] = 'a4.5'
    map_d['Y'] = 'a8'
    callgmt( 'grdimage', cont_name, map_d, '>>', ps )

    C = cross_section_dir + 'cont.cpt'
    cmd = '-Ba1:"Continental type (stencil value)": -D6/7.5/2.5/0.1h -C%(C)s -K -O' % vars()
    callgmt( 'psscale', cmd, '', '>>', ps )

    del map_d['B']
    del map_d['C']
    map_d['m'] = ' ' 
    map_d['W'] = '5,black'
    callgmt( 'psxy', proj_name, map_d, '>>', ps )
    del map_d['m']
    del map_d['W']
    map_d['G'] = 'purple'
    map_d['S'] = 'c0.05'
    callgmt( 'psxy', lon_markers, map_d, '>>', ps )
    del map_d['G']
    del map_d['S']

    # end plot maps #


    # plot cross-sections

    # temperature cross-section
    psbase_d = opts_d.copy()
    psbase_d['B'] = 'a10/500::WsNe'
    psbase_d['J'] = 'Pa6/' + str(pao) + 'z'
    psbase_d['R'] = psbase_R
    psbase_d['X'] = 'a1.25'
    psbase_d['Y'] = 'a5.25'
    callgmt( 'psbasemap', '', psbase_d, '>>', ps ) 

    opts_d['C'] = cross_section_dir + 'temp.cpt'
    opts_d['J'] = 'Pa6/' + str(pao)
    opts_d['R'] = R_ann
    opts_d['X'] = 'a1.25'
    opts_d['Y'] = 'a5.25'
    callgmt( 'grdimage', grid_names[0], opts_d, '>>', ps )

    # profile of lith_age_depth on this cross-section
    del opts_d['C']
    opts_d['W'] = '3,black,-'
    callgmt( 'psxy', xy, opts_d, '>>', ps )
    del opts_d['W']
    opts_d['G'] = 'purple'
    opts_d['N'] = ' '
    opts_d['S'] = 'c0.06'
    callgmt( 'psxy', lon_markers_ann, opts_d, '>>', ps )
    del opts_d['G']
    del opts_d['N']
    del opts_d['S']

    stdin = '1 6.25 12 0 4 ML CitcomS\n'
    stdin += '7.5 6.25 12 0 4 MR Temp\nEOF'
    callgmt( 'pstext', '', pstext_d, '<< EOF >>', ps + '\n' + stdin )

    C = cross_section_dir + 'temp.cpt'
    cmd = '-Ba0.2f0.1 -D4.25/5.7/2.5/0.1h -C%(C)s -K -O' % vars()
    callgmt( 'psscale', cmd, '', '>>', ps )


    # idealized lith temperature cross-section
    psbase_d['Y'] = 'a3.75'
    callgmt( 'psbasemap', '', psbase_d, '>>', ps )

    opts_d['C'] = cross_section_dir + 'temp.cpt'
    opts_d['Y'] = 'a3.75'
    callgmt( 'grdimage', grid_names[1], opts_d, '>>', ps )
    del opts_d['C']

    # profile of lith_age_depth on this cross-section
    opts_d['W'] = '3,black,-'
    callgmt( 'psxy', xy, opts_d, '>>', ps )
    del opts_d['W']

    opts_d['G'] = 'purple'
    opts_d['N'] = ' '
    opts_d['S'] = 'c0.06'
    callgmt( 'psxy', lon_markers_ann, opts_d, '>>', ps )
    del opts_d['G']
    del opts_d['N']
    del opts_d['S']

    stdin = '1 4.75 12 0 4 ML Idealised\n'
    stdin += '7.5 4.75 12 0 4 MR Temp\nEOF'
    callgmt( 'pstext', '', pstext_d, '<< EOF >>', ps + '\n' + stdin )

    C = cross_section_dir + 'temp.cpt'
    cmd = '-Ba0.2f0.1 -D4.25/4.2/2.5/0.1h -C%(C)s -K -O' % vars()
    callgmt( 'psscale', cmd, '', '>>', ps )



    # contours plot
    psbase_d['Y'] = 'a2.25'
    callgmt( 'psbasemap', '', psbase_d, '>>', ps )
    opts_d['Y'] = 'a2.25'
    opts_d['C'] = cross_section_dir + 'temp.cont'
    opts_d['W'] = '3,red'
    callgmt( 'grdcontour', grid_names[0], opts_d, '>>', ps )
    opts_d['W'] = '3,green'
    callgmt( 'grdcontour', grid_names[1], opts_d, '>>', ps )
    del opts_d['C']
    del opts_d['W']

    opts_d['G'] = 'purple'
    opts_d['N'] = ' '
    opts_d['S'] = 'c0.06'
    callgmt( 'psxy', lon_markers_ann, opts_d, '>>', ps )
    del opts_d['G']
    del opts_d['N']
    del opts_d['S']

    stdin = '1 3.25 12 0 4 ML Contours\n'
    stdin += '7.5 3.25 12 0 4 MR Temp\nEOF'
    callgmt( 'pstext', '', pstext_d, '<< EOF >>', ps + '\n' + stdin )

    # difference of temperature fields (relative)
    psbase_d['Y'] = 'a0.75'
    callgmt( 'psbasemap', '', psbase_d, '>>', ps )
    opts_d['C'] = cross_section_dir + 'diff.cpt'
    opts_d['Y'] = 'a0.75'
    callgmt( 'grdimage', grid_names[2], opts_d, '>>', ps )
    del opts_d['C']

    opts_d['G'] = 'purple'
    opts_d['N'] = ' '
    opts_d['S'] = 'c0.06'
    callgmt( 'psxy', lon_markers_ann, opts_d, '>>', ps )
    del opts_d['G']
    del opts_d['N']
    del opts_d['S']

    C = cross_section_dir + 'diff.cpt'
    cmd = '-Ba5f1 -D4.25/1.2/2.5/0.1h -C%(C)s -K -O' % vars()
    callgmt( 'psscale', cmd, '', '>>', ps )

    stdin = '1 1.75 12 0 4 ML Delta (\045)\n'
    stdin += '7.5 1.75 12 0 4 MR Temp\nEOF'
    #stdin += '4.25 0.6 12 0 4 MC Note: No assimilation regions are shown in BLACK\nEOF'
    callgmt( 'pstext', '', pstext_d, '<< EOF >>', ps + '\n' + stdin )


    Core_GMT.end_postscript( ps )

    # clean up temporary files
    Core_Util.remove_files( rm_list )

#=====================================================================
#=====================================================================
#=====================================================================

if __name__ == "__main__":

    main()

#=====================================================================
#=====================================================================
#=====================================================================
