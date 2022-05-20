#!/usr/bin/env python
#=====================================================================
#                  Python Scripts for Data Assimilation 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner, Sabin Zahirovic
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''Core_GMT6.py provides a generalized interface to the various GMT6 programs, as well as a few specialized plotting functions '''
#=====================================================================
#=====================================================================
import os, sys, subprocess
import numpy as np
import Core_Util

#====================================================================
verbose = True
#====================================================================
#====================================================================
#====================================================================
def usage():

    sys.exit()

#====================================================================
#====================================================================
#====================================================================
def apos( X ):
    '''Convert argument to GMT absolute position'''

    return 'a'+str(X)

#====================================================================
#====================================================================
#====================================================================
def callpstext( text_l=[], opts='', redirect='', out='' ):

    '''Wrapper for pstext to read stdin.

    text_l: list of GMT text entries.'''

    text_cmd = '\n' + '\n'.join( text_l )
    redirect2 = '<< EOF ' + redirect
    out2 = out + text_cmd + '\nEOF'
    callgmt( 'pstext', '', opts, redirect2, out2 )

#====================================================================
#====================================================================
#====================================================================
def callgmt( gmtcmd, arg, opts='', redirect='', out='' ):

    '''Call a generic mapping tools (GMT) command.

The arguments to this function are all single strings, and typically 
constructed from parameters in client code. Only gmtcmd is required,
all other arguments are optional, depending on the GMT command to call.

gmtcmd : the actual GMT command to call, almost always a single string value;
arg : the required arguments for the command. 
opts : the optional arguments for the command.
redirect : the termnal redirect symbol, usually '>', sometimes a pipe '|' or input '<'.
out : the output file name.

'''

    # build list of commands
    cmd_list = ['gmt ' + gmtcmd]

    # (required) arguments
    if arg: cmd_list.append( arg )

    # options
    if opts:
        cmd_list.extend('-'+str(k)+str(v) for k, v in opts.items())

    # redirect
    if redirect:
        cmd_list.append(redirect)

    # out file
    if out:
        cmd_list.append(out)

    # create one string
    cmd = ' '.join(cmd_list)
    if verbose: print( Core_Util.now(), cmd )

    # capture output (returned as bytes)
    p = subprocess.check_output( cmd, shell=True )

    # convert bytes output to string and remove
    # newline character
    s = bytes.decode(p).rstrip()

    return s

#====================================================================
#====================================================================
#====================================================================
def start_postscript( ps ):

    '''Start a postscript'''

    if verbose: print( Core_Util.now(), 'start_postscript:' )

    arg = 'PS_MEDIA letter PROJ_LENGTH_UNIT inch '
    arg += 'MAP_ORIGIN_X 0 MAP_ORIGIN_Y 0'
    callgmt( 'gmtset', arg, '', '', '' )
    opts = {'K':'','T':'','R':'0/1/0/1','J':'x1.0'}
    callgmt( 'psxy', '', opts, '>', ps )
    opts = {'K':'', 'O':''}

    return opts

#====================================================================
#====================================================================
#====================================================================
def end_postscript( ps ):

    '''End a postscript'''

    if verbose: print( Core_Util.now(), 'end_postscript:' )

    opts = {'T':'','O':'','R':'0/1/0/1','J':'x1.0'}
    callgmt( 'psxy', '', opts, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_age_grid_mask( opts_d, ps, age ):

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    age_grid_dir = geoframe_d['age_grid_mask_dir']
    age_grid_prefix = geoframe_d['age_grid_mask_prefix']

    arg = age_grid_dir + '/' + age_grid_prefix
    arg += '%(age)s.grd' % vars()
    cmd = '%(arg)s -Rg -S' % vars()
    callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360
    # Jono edit: check for a .nc grid if the .grd grid doesn't exist
    if not os.path.exists( arg ):
        #arg += '%(age)s.0Ma.nc' % vars()
        arg += '%(age)s.nc' % vars()
        cmd = '%(arg)s -Rg -S' % vars()
        callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360
    if not os.path.exists( arg ):
        arg += '%(age)s.0Ma.nc' % vars()
        cmd = '%(arg)s -Rg -S' % vars()
        callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360


    if os.path.exists( arg ):
        callgmt( 'grdimage', arg, opts_d, '>>', ps )


#====================================================================
#====================================================================
#====================================================================
def plot_age_grid_no_mask( opts_d, ps, age ):

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    age_grid_dir = geoframe_d['age_grid_no_mask_dir']
    age_grid_prefix = geoframe_d['age_grid_no_mask_prefix']

    arg = age_grid_dir + '/' + age_grid_prefix
    arg += '%(age)s.grd' % vars()
    cmd = '%(arg)s -Rg -S' % vars()
    callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360
    # Jono edit: check for a .nc grid if the .grd grid doesn't exist
    if not os.path.exists( arg ):
        arg += '%(age)s.nc' % vars()
        cmd = '%(arg)s -Rg -S' % vars()
        callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360

    if os.path.exists( arg ):
        callgmt( 'grdimage', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_age_grid_continent( opts_d, ps, age ):

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    age_grid_dir = geoframe_d['age_grid_cont_dir']
    age_grid_prefix = geoframe_d['age_grid_cont_prefix']

    arg = age_grid_dir + '/' + age_grid_prefix
    arg += '%(age)s.grd' % vars()
    cmd = '%(arg)s -Rg -S' % vars()
    callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360
    # Jono edit: check for a .nc grid if the .grd grid doesn't exist
    if not os.path.exists( arg ):
        arg += '%(age)s.nc' % vars()
        cmd = '%(arg)s -Rg -S' % vars()
        callgmt( 'grdedit', cmd ) # Make sure the longitude is 0/360

    if os.path.exists( arg ):
        callgmt( 'grdimage', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_coastline( opts_d, ps, age ):

    '''Plot GPlates coastlines.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_coast_dir']
    arg = gplates_line_dir + '/reconstructed_%(age)s.00Ma.xy' % vars()
    if os.path.exists( arg ):
        callgmt( 'psxy', arg, opts_d, '>>', ps ) 

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_plate_polygon( opts_d, ps, age ):

    '''Plot GPlates plate polygons.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    arg = gplates_line_dir + \
        '/topology_platepolygons_%(age)s.00Ma.xy' % vars()
    if os.path.exists( arg ):
        callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_slab_polygon( opts_d, ps, age ):

    '''Plot GPlates slab polygons.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    arg = gplates_line_dir + \
        '/topology_slab_polygons_%(age)s.00Ma.xy' % vars()
    if os.path.exists( arg ):
        callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_ridge_and_transform( opts_d, ps, age ):

    '''Plot GPlates ridges and transforms.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    arg = gplates_line_dir + \
        '/topology_ridge_transform_boundaries_%(age)s.00Ma.xy' % vars()
    if os.path.exists( arg ):
        callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_transform( opts_d, ps, age ):

    '''Plot GPlates transforms.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    linefile = gplates_line_dir + \
        '/topology_ridge_transform_boundaries_%(age)s.00Ma.xy' % vars()

    if not os.path.exists( linefile ): return

    # process to write out only ">Transform" data to temporary file
    # for plotting
    infile = open( linefile, 'r' )
    lines = infile.readlines()
    infile.close()

    outname = 'ridges.xy'
    outfile = open( outname, 'w' )

    flag = 0

    for line in lines:
        if line.startswith('>'):
            flag = 0 # reset
        if line.startswith('>Transform'):
            flag = 1 # write out all subsequent lines
        if flag:
            outfile.write( line )

    outfile.close()

    callgmt( 'psxy', outname, opts_d, '>>', ps )

    Core_Util.remove_files( [outname] )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_leading_edge( opts_d, ps, age,
    linestyle = 'sawtooth' ):

    '''Plot GPlates leading edge.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_subduction_prefix = gplates_line_dir + \
        '/topology_'

    for subtype in [ 'slab_edges_leading']:
        for polarity in [ 'sL', 'sR' ]:
            symbarg = polarity[-1].lower()
            suffix = '_%(polarity)s_%(age)0.2fMa.xy' % vars()
            arg = gplates_subduction_prefix + subtype + suffix
            if os.path.exists( arg ):
                if linestyle == 'sawtooth':
                    S = 'f0.2i/0.05i+%(symbarg)s+t' % vars()
                    arg += ' -S%(S)s' % vars()
                callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_no_assimilation_stencil( opts_d, ps, age ):

    '''Plot GPlates no assimilation stencils.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    no_ass_dir = geoframe_d['no_ass_dir']
    arg = no_ass_dir + \
        '/topology_network_polygons_%(age)0.2fMa.xy' % vars()
    if os.path.exists( arg ):
        callgmt( 'psxy', arg, opts_d, '>>', ps )
    else:
       print('cannot find file:', arg )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_subduction( opts_d, ps, age,
    linestyle = 'sawtooth' ):

    '''Plot GPlates subduction zones.'''

    geoframe_d = Core_Util.parse_geodynamic_framework_defaults()
    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_subduction_prefix = gplates_line_dir + \
        '/topology_'

    for subtype in [ 'subduction' ]: #, 'network_subduction' ]:
        for polarity in [ 'sL', 'sR' ]:
            symbarg = polarity[-1].lower()
            suffix = '_%(polarity)s_%(age)0.2fMa.xy' % vars()
            arg = gplates_subduction_prefix + subtype + \
                '_boundaries' + suffix
            if os.path.exists( arg ):
                if linestyle == 'sawtooth':
                    S = 'f0.2i/0.05i+%(symbarg)s+t' % vars()
                    arg += ' -S%(S)s' % vars()
                callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_velocity_scale( opts_d, velocity_scale, arrow_length_inches,
    ps ):

    # user should set GMT options in opts_d dictionary
    # velocity scale is one inch, so adjust according to user-specified
    # length
    velocity_scale /= 1 / float( arrow_length_inches )

    stdin = '''# velocity vector
S 0.125 v %(arrow_length_inches)s/0.015/0.06/0.05 0/0/0''' % vars()
    stdin += ''' 1,black 0.27i %(velocity_scale).0f cm/yr\n''' % vars()
    stdin += 'EOF'

    callgmt( 'pslegend', '', opts_d, '<< EOF >>', ps + '\n' + stdin)

#====================================================================
#====================================================================
#====================================================================
def ps2raster( ps, opts_d={} ):

    # if a dictionary is also passed, use these values
    if opts_d:
        callgmt( 'ps2raster', ps, opts_d, '', '' )
    # otherwise, if just a ps, then convert to publication quality eps
    # with a tight crop
    else:
        arg = '%(ps)s -A -Te -E300' % vars()
        callgmt( 'ps2raster', arg, '', '' ,'' )

#====================================================================
#====================================================================
#====================================================================
def get_T_from_minmax(xyz_filename) :
    ''' get a -T value from minmax on a xyz file'''
    cmd = 'gmt minmax -C %(xyz_filename)s' % vars()
    s = subprocess.check_output( cmd, shell=True, universal_newlines=True)
    if verbose: print( Core_Util.now(), cmd )
    l = s.split()
    min = float(l[4])
    max = float(l[5])

    if   max >=  10000000 : dt =  1000000.
    elif max >=    100000 : dt =     1000.
    elif max >=      1000 : dt =      100.
    elif max >=       0.1 : dt =         .01 # Jono changed the >= value from 1 to 0.1
    else                  : dt =        1.0

    # Jono edit: make sure the min and max values aren't the same
    if max - min <= dt :
        max = max + dt 
        min = min - dt 

    T = '-T%(min)s/%(max)s/%(dt)s' % vars()
    if verbose: print( Core_Util.now(), 'T =', T )

    return T
#====================================================================
#====================================================================
def get_T_from_grdinfo(grid_filename):
    '''get a -T value from grdinfo on a grid file'''

    cmd = 'grdinfo -C %(grid_filename)s' % vars()
    s = subprocess.check_output( cmd, shell=True, universal_newlines=True)
    if verbose: print( Core_Util.now(), cmd )
    l = s.split()
    min = float(l[5])
    max = float(l[6])

    if   max >=  10000000 : dt =  1000000.
    elif max >=    100000 : dt =     1000.
    elif max >=      1000 : dt =      100.
    elif max >=       0.1 : dt =         .01 # Jono changed the >= value from 1 to 0.1
    else                  : dt =        1.0

    # Jono edit: make sure the min and max values aren't the same
    if max == min :
        max = max + dt 
        min = min - dt 

    T = '-T%(min)s/%(max)s/%(dt)s' % vars()
    if verbose: print( Core_Util.now(), 'T =', T )

    return T
#====================================================================
def plot_grid( grid_filename, xy_filename = None, R_value = 'g', T_value = '-T0/1/.1', J_value = 'X8/5', C_value = 'polar' ):
    '''simple function to make a test plot'''
    global verbose
    verbose = True 

    # postscript name
    ps = grid_filename.rstrip('.nc') + '.ps'

    # page orientation must be set before start_postscript()
    arg = 'PAGE_ORIENTATION landscape'
    callgmt( 'gmtset', arg, '', '', '' )

    # start postscript
    # the returned dictionary has 'O' and 'K' set
    opts = start_postscript( ps )

    # psbasemap 2
    opts['X'] = apos(3)
    opts['Y'] = apos(3)
    opts['R'] = R_value # either regional : '0/57/-14/14' ; or global: 'g'
    opts['B'] = 'a30'
    opts['J'] = 'X5/3' #'R0/6'
    callgmt( 'psbasemap', '', opts, '>>', ps )

    # create a cpt for this grid
    cpt_file = grid_filename.replace('.nc', '.cpt')
    
    cmd = '-Cpolar ' + T_value 
    callgmt( 'makecpt', cmd, '', '>', cpt_file )

    # grdimage
    opts['C'] = cpt_file
    callgmt( 'grdimage', grid_filename, opts, '>>', ps )

    # psxy
    del opts['C']
    opts['m'] = ' ' 
    try:
        callgmt( 'psxy', xy_filename, opts, '>>', ps )
    except:
        print("the xy file in plot_grid was not plotted")

    # end postscript
    end_postscript( ps )

    # create a .png image file
    #cmd = 'convert -resize 300% -rotate 90 ' + ps + ' ' + ps.replace('.ps', '.png')
    cmd = 'convert -rotate 90 ' + ps + ' ' + ps.replace('.ps', '.png')
    if verbose: print( Core_Util.now(), cmd )
    # call
    subprocess.call( cmd, shell=True )

#====================================================================
#====================================================================
def test( argv ):
    '''self test'''
    global verbose
    verbose = True 

    # -------------------------------------------------
    # proposed new way of wrapping GMT using python 3.0
    # example plotting script follows
    # -------------------------------------------------

    # postscript name
    ps = 'test.ps'

    # page orientation must be set before start_postscript()
    arg = 'PAGE_ORIENTATION landscape'
    callgmt( 'gmtset', arg, '', '', '' )

    # start postscript
    # the returned dictionary has 'O' and 'K' set
    opts = start_postscript( ps )

    # psbasemap 1
    opts['B'] = 'a1'
    opts['J'] = 'X2.0'
    opts['R'] = '0/10/0/10'
    opts['X'] = apos(0.5)
    opts['Y'] = apos(1)
    callgmt( 'psbasemap', '', opts, '>>', ps )

    # psbasemap 2
    opts['X'] = apos(3)
    opts['Y'] = apos(3)
    opts['R'] = 'g'
    opts['B'] = 'a60'
    opts['J'] = 'H0/6'
    callgmt( 'psbasemap', '', opts, '>>', ps )

    # pscoast
    del opts['B']
    opts['I'] = 1
    opts['N'] = 2
    callgmt( 'pscoast', '', opts, '>>', ps )

    # end postscript
    end_postscript( ps )

#====================================================================
#====================================================================
#====================================================================
if __name__ == "__main__":
    import Core_GMT5

    if len( sys.argv ) > 1:
        # process sys.arv as file names for testing 
        #test( sys.argv )

        # test functions directly
        print ('T =', get_T_from_minmax( sys.argv[1] ) )

    else:
        # print module documentation and exit
        help(Core_GMT5)

#====================================================================
#====================================================================
#====================================================================
