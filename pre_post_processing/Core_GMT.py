#!/usr/bin/env python
#=====================================================================
#                  Python Scripts for Data Assimilation 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''Core_GMT.py provides a generalized interface to the various GMT programs, as well as a few specialized plotting functions '''
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
def callgmt( gmtcmd, arg, opts='', redirect='', out='' ):

    '''Call a generic mapping tools (GMT) command.

The arguments to this function are all single strings, and typically 
constructed from parameters in client code. Only gmtcmd is required,
all other arguments are optional, depending on the GMT command to call.

gmtcmd : the actual GMT command to call, almost always a single string value;
arg : the required arguments for the command (string). 
opts : the optional arguments for the command (in a dictionary).
redirect : the termnal redirect symbol, usually '>', sometimes a pipe '|' or input '<'.
out : the output file name.

'''

    # build list of commands
    cmd_list = ['gmt', gmtcmd]

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

    # always report on calls to GMT for log files 
    print( Core_Util.now(), cmd )

    # capture output (returned as bytes)
    p = subprocess.check_output( cmd, shell=True )

    # convert bytes output to string
    s = bytes.decode(p)

    return s

#====================================================================
#====================================================================
#====================================================================
def start_postscript( ps ):

    '''Start a postscript'''

    if verbose: print( Core_Util.now(), 'start_postscript:' )

    arg = 'PAPER_MEDIA letter MEASURE_UNIT inch '
    arg += 'X_ORIGIN 0 Y_ORIGIN 0'
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
def plot_gplates_coastline( geoframe_d, opts_d, ps, age, W ):

    '''Plot GPlates coastlines.'''

    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_coast_dir']
    gplates_coastline = gplates_line_dir + \
        '/reconstructed_%(age)s.00Ma.xy' % vars()
    arg = gplates_coastline + ' -m -W%(W)s' % vars()
    callgmt( 'psxy', arg, opts_d, '>>', ps ) 

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_slab_polygon( geoframe_d, opts_d, ps, age, W ):

    '''Plot GPlates slab polygons.'''

    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_slab_polygon = gplates_line_dir + \
        '/topology_slab_polygons_%(age)s.00Ma.xy' % vars()
    arg = gplates_slab_polygon + ' -m -W%(W)s' % vars()
    callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_ridge_and_transform( geoframe_d, opts_d, ps, age, W ):

    '''Plot GPlates ridges and transforms.'''

    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_ridge_transform = gplates_line_dir + \
        '/topology_ridge_transform_boundaries_%(age)s.00Ma.xy' % vars()
    arg = gplates_ridge_transform + ' -m -W%(W)s' % vars()
    callgmt( 'psxy', arg, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_line_subduction( geoframe_d, opts_d, ps, age, W ):

    '''Plot GPlates subduction zones with a line.'''

    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_subduction_prefix = gplates_line_dir + \
        '/topology_'
    arg = ' -m -W%(W)s' % vars()

    for subtype in [ 'subduction', 'network_subduction' ]:
        for polarity in [ 'sL', 'sR' ]:
            symbarg = polarity[-1].lower()
            suffix = '_%(polarity)s_%(age)0.2fMa.xy' % vars()
            line_data = gplates_subduction_prefix + subtype + \
                '_boundaries' + suffix
            arg2 = line_data + arg
            callgmt( 'psxy', arg2, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_sawtooth_leading_edge( geoframe_d, opts_d, ps, age, W, G ):

    '''Plot GPlates leading edge with a sawtooth line.'''

    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_subduction_prefix = gplates_line_dir + \
        '/topology_'
    arg = ' -m -W%(W)s -G%(G)s' % vars()

    for subtype in [ 'slab_edges_leading']:
        for polarity in [ 'sL', 'sR' ]:
            symbarg = polarity[-1].lower()
            suffix = '_%(polarity)s_%(age)0.2fMa.xy' % vars()
            line_data = gplates_subduction_prefix + subtype + suffix
            S = 'f0.2i/0.05i%(symbarg)st' % vars()
            arg2 = line_data + arg + ' -S%(S)s' % vars()
            callgmt( 'psxy', arg2, opts_d, '>>', ps )

#====================================================================
#====================================================================
#====================================================================
def plot_gplates_sawtooth_subduction( geoframe_d, opts_d, ps, age, W, G ):

    '''Plot GPlates subduction zones with a sawtooth line.'''

    age = int( age ) # ensure integer
    gplates_line_dir = geoframe_d['gplates_line_dir']
    gplates_subduction_prefix = gplates_line_dir + \
        '/topology_'
    arg = ' -m -W%(W)s -G%(G)s' % vars()

    for subtype in [ 'subduction', 'network_subduction' ]:
        for polarity in [ 'sL', 'sR' ]:
            symbarg = polarity[-1].lower()
            suffix = '_%(polarity)s_%(age)0.2fMa.xy' % vars()
            line_data = gplates_subduction_prefix + subtype + \
                '_boundaries' + suffix
            S = 'f0.2i/0.05i%(symbarg)st' % vars()
            arg2 = line_data + arg + ' -S%(S)s' % vars()
            callgmt( 'psxy', arg2, opts_d, '>>', ps )

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
def get_T_from_minmax(xyz_filename) :
    ''' get a -T value from minmax on a xyz file'''
    cmd = 'minmax -C %(xyz_filename)s' % vars()
    s = subprocess.check_output( cmd, shell=True, universal_newlines=True)
    if verbose: print( Core_Util.now(), cmd )
    l = s.split()
    min = float(l[4])
    max = float(l[5])

    # FIXME: stop gap measure 
    if min == max : 
        print( Core_Util.now(), 'get_T_from_minmax: WARNING: min == max: min =', min, '; max =', max )
        min = 0.0
        max = 1.0
        dt  = 0.01
        T = '-T%(min)s/%(max)s/%(dt)s' % vars()
        if verbose: print( Core_Util.now(), 'get_T_from_minmax: T =', T )
        return T

    if   max >=  10000000    : dt =  1000000.0
    elif max >=    100000    : dt =     1000.0
    elif max >=      1000    : dt =      100.0
    elif max >=         1    : dt =         .1
    elif max >=         0.1  : dt =         .01
    elif max >=         0.01 : dt =         .001
    else                     : dt =        1.0

    T = '-T%(min)s/%(max)s/%(dt)s' % vars()
    if verbose: print( Core_Util.now(), 'get_T_from_minmax: T =', T )
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

    # FIXME: stop gap measure 
    if min == max : 
        min = 0.0
        max = 1.0
        dt  = 0.01
        T = '-T%(min)s/%(max)s/%(dt)s' % vars()
        if verbose: print( Core_Util.now(), 'get_T_from_grdinfo: WARNING: min==max, setting T =', T )
        return T

    if   max >=  10000000    : dt =  1000000.0
    elif max >=    100000    : dt =     1000.0
    elif max >=      1000    : dt =      100.0
    elif max >=         1    : dt =         .1
    elif max >=         0.1  : dt =         .01
    elif max >=         0.01 : dt =         .001
    else                     : dt =        1.0

    T = '-T%(min)s/%(max)s/%(dt)s' % vars()
    if verbose: print( Core_Util.now(), 'get_T_from_grdinfo: T =', T )
    return T
#====================================================================
def plot_grid( grid_filename, xy_filename = None, R_value = 'g', T_value = '-T0/1/.1', J_value = 'X8/5', C_value = 'polar'):
    '''simple function to make a test plot'''
    global verbose
    verbose = True 

    # postscript name
    ps = grid_filename.rstrip('.grd') + '.ps'

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
    opts['J'] = J_value
    opts['B'] = 'a30'
    callgmt( 'psbasemap', '', opts, '>>', ps )

    # create a cpt for this grid
    cpt_file = grid_filename.replace('.grd', '.cpt')
    
    #cmd = '-Cpolar ' + T_value 
    cmd = '-C' + C_value + ' ' + T_value
    callgmt( 'makecpt', cmd, '', '>', cpt_file )

    # grdimage
    opts['C'] = cpt_file
    callgmt( 'grdimage', grid_filename, opts, '>>', ps )

    # psxy
    del opts['C']
    opts['m'] = ' ' 
    if xy_filename :
        callgmt( 'psxy', xy_filename, opts, '>>', ps )

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
    import Core_GMT

    if len( sys.argv ) > 1:
        # process sys.arv as file names for testing 
        #test( sys.argv )

        # test functions directly
        print ('T =', get_T_from_minmax( sys.argv[1] ) )

    else:
        # print module documentation and exit
        help(Core_GMT)

#====================================================================
#====================================================================
#====================================================================
