#!/usr/bin/env python
#=====================================================================
#                 Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                       AUTHORS: Dan J. Bower
#
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#=====================================================================
import sys
import numpy as np
import Core_GMT
import subprocess
from collections import defaultdict
from Core_Util import now
from Core_GMT import callgmt
#=====================================================================
verbose = True
#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    '''print usage message and exit'''

    print('''plot_convergence.py log_filename start_timestep end_timestep
''')

    sys.exit(0)

#=====================================================================
#=====================================================================
#=====================================================================
def main():

    if len(sys.argv) != 4:
        usage()

    log_filename = sys.argv[1]
    start_timestep = sys.argv[2]
    end_timestep = sys.argv[3]

    data_d = read_log_file_and_make_dictionary( log_filename )

    make_plot( data_d, start_timestep, end_timestep )

#=====================================================================
#=====================================================================
#=====================================================================
def make_plot( data_d, start_timestep, end_timestep ):

    '''make postscript using GMT commands'''

    figure_width = 3.0
    figure_height = 3

    # start postscript
    callgmt( 'gmtset', 'HEADER_FONT_SIZE', '', '', '12' )
    callgmt( 'gmtset', 'HEADER_OFFSET', '', '', '-0.1' )
    callgmt( 'gmtset', 'PAGE_ORIENTATION', '', '', 'portrait' )
    callgmt( 'gmtset', 'LABEL_FONT_SIZE', '', '', '12' )
    callgmt( 'gmtset', 'LABEL_FONT', '', '', '4' )
    callgmt( 'gmtset', 'LABEL_OFFSET', '', '', '0.02' )
    callgmt( 'gmtset', 'ANNOT_FONT_SIZE_PRIMARY', '', '', '10p' )
    callgmt( 'gmtset', 'ANNOT_FONT_PRIMARY', '', '', '4' )

    ps = 'convergence.ps'

    options_d = Core_GMT.start_postscript( ps )

    # plot convergence criteria
    options_d['X'] = 'a0.83'
    options_d['Y'] = 'a5.75'
    options_d['J'] = 'X%(figure_width)s/%(figure_height)sl' % vars()
    options_d['R'] = '%(start_timestep)s/%(end_timestep)s/1E-5/1' % vars()
    options_d['B'] = 'a100:"time step":/a1f3p:"convergence"::."accuracy":WeSn'
    callgmt('psbasemap', '', options_d, '>>', ps )
    del options_d['B']

    colours = ['red','blue','green']
    for nn, entry in enumerate( ['div/v','dv/v','dp/p'] ):
        name = entry + '_per_step'
        y = data_d[name]
        x = np.arange( np.size(data_d[name]) )
        np.savetxt( 'temp.xy', np.column_stack( (x,y) ) )
        cmd = 'temp.xy -W3,' + colours[nn] 
        callgmt( 'psxy', cmd, options_d, '>>', ps )

    # plot time information
    options_d['X'] = 'a4.83'
    options_d['Y'] = 'a5.75'
    options_d['J'] = 'X%(figure_width)s/%(figure_height)s' % vars()
    options_d['R'] = '%(start_timestep)s/%(end_timestep)s/0/1500' % vars()
    options_d['B'] = 'a100:"time step":/a100:"solver time (s)"::."solver time":WeSn' 
    callgmt('psbasemap', '', options_d, '>>', ps )
    del options_d['B']

    for nn, entry in enumerate( ['time'] ):
        name = entry + '_per_step'
        y = data_d[name]
        x = np.arange( np.size(data_d[name]) )
        np.savetxt( 'temp.xy', np.column_stack( (x,y) ) )
        cmd = 'temp.xy -W3,black'
        callgmt( 'psxy', cmd, options_d, '>>', ps )

    # plot cumulative time information
    options_d['X'] = 'a0.83'
    options_d['Y'] = 'a1.5'
    options_d['J'] = 'X%(figure_width)s/%(figure_height)s' % vars()
    options_d['R'] = '%(start_timestep)s/%(end_timestep)s/0/20' % vars()
    options_d['B'] = 'a100:"time step":/a5f1:"cumulative solver time (hr)"::."cumulative solver time":WeSn'
    callgmt('psbasemap', '', options_d, '>>', ps )
    del options_d['B']

    for nn, entry in enumerate( ['time_cumsum'] ):
        name = entry + '_per_step_hours'
        y = data_d[name]
        x = np.arange( np.size(data_d[name]) )
        np.savetxt( 'temp.xy', np.column_stack( (x,y) ) )
        cmd = 'temp.xy -W3,black'
        callgmt( 'psxy', cmd, options_d, '>>', ps )

    # plot number of iterations
    options_d['X'] = 'a4.83'
    options_d['Y'] = 'a1.5'
    options_d['J'] = 'X%(figure_width)s/%(figure_height)s' % vars()
    options_d['R'] = '%(start_timestep)s/%(end_timestep)s/0/102' % vars()
    options_d['B'] = 'a100:"time step":/a25:"iterations"::."iterations":WeSn' 
    callgmt('psbasemap', '', options_d, '>>', ps )
    del options_d['B']

    for nn, entry in enumerate( ['iteration'] ):
        name = entry + '_per_step'
        y = data_d[name]
        x = np.arange( np.size(data_d[name]) )
        np.savetxt( 'temp.xy', np.column_stack( (x,y) ) ) 
        cmd = 'temp.xy -W3,black'
        callgmt( 'psxy', cmd, options_d, '>>', ps )

    subprocess.call( 'rm temp.xy', shell=True )

    # end postscript
    Core_GMT.end_postscript( ps )

#=====================================================================
#=====================================================================
#=====================================================================
def read_log_file_and_make_dictionary( log_filename ):

    '''process a CitcomS *.log file and populate a dictionary with
       lists containing the data (convergence, solver time etc.)'''

    # read log file
    log_file = open(log_filename,'r')
    lines = log_file.readlines()
    log_file.close()

    # store lists in a (default) dict
    # for those unfamiliar with a defaultdict, it acts like a
    # dictionary but will not throw a KeyError if the key does not
    # exist.  Instead, it will create the key for you with a given type
    # (in this case, a list).  This saves initializing a bunch of lists
    # that may not be used.
    data_d = defaultdict( list )

    # loop over all lines in log file
    for line in lines:

        if line.startswith('(') and line[1].isdigit():
            # iteration number
            line_l = line.split('(')
            line_l = line_l[1].split(')')
            iter_num = int(line_l[0])
            data_d['iteration'].append( iter_num )

            # time
            line_l = line.split()
            time = float(line_l[1])
            data_d['time'].append( time )

            # div/v, dv/v, and dp/p
            for criteria in ['div/v','dv/v','dp/p']:
                line_l = line.split( '%(criteria)s=' % vars() )
                line_l = line_l[1].split(' ')
                criteria_data = float(line_l[0])
                data_d[criteria].append( criteria_data )

        # starting age and current age
        if 'Starting Age' in line:
            line_l = line.split('Starting Age')
            line_l = line_l[1].split(',')
            line_l = line_l[0].split('=')
            starting_age = float(line_l[1])
            data_d['starting_age'].append( starting_age )

        if 'Current Age' in line:
            line_l = line.split('Current Age')
            # need to remove '='
            line_l = line_l[-1].split('=')
            current_age = float(line_l[1])
            data_d['current_age'].append( current_age )


    # convert all lists in data_d to numpy arrays
    for key, item in data_d.items():
        data_d[key] = np.array(item)
        if verbose: print( now(), 'array=', key, ', size=', np.size(data_d[key]) )

    # some post processing
    # runtime (Myr)
    data_d['runtime'] = data_d['starting_age'] \
                                 - data_d['current_age']

    # ==================================
    # quantities PER time step
    # ==================================

    # *** determine indices for the last iteration for each time step ***
    # find indices of zero iterations
    last_iter_for_step_indices = np.where( data_d['iteration'] == 0 )[0]
    # minus one will give us the maximum number for the previous time step
    last_iter_for_step_indices -= 1
    # remove first entry which is now meaningless (it is negative)
    last_iter_for_step_indices = last_iter_for_step_indices[1:]
    # add in last index to catch final iteration
    last_iter_for_last_step_index = last_iter_for_step_indices[-1] + data_d['iteration'][-1] + 1
    last_iter_for_step_indices = np.append( last_iter_for_step_indices, last_iter_for_last_step_index )
    no_of_time_steps = np.size(last_iter_for_step_indices)
    if verbose: print( now(), 'number of time steps=', no_of_time_steps )

    # now slice the data
    for criteria in ['iteration','time','div/v','dv/v','dp/p']:
        name = criteria+'_per_step'
        data_d[name] = data_d[criteria][last_iter_for_step_indices]
        if verbose: print( now(), 'array= %(name)s, size=' % vars(), \
            np.size(data_d[name]) )

    for criteria in ['current_age', 'runtime', 'starting_age']:
        name = criteria+'_per_step'
        # unfortunately np.unique also sorts the data which we don't want
        # hence the added complexity with the call
        indexes = np.unique( data_d[criteria], return_index=True )[1]
        data_d[name] = [data_d[criteria][index] for index in sorted(indexes)]
        data_d[name] = np.array( data_d[name] )
        # if the log file does not terminate cleanly then sometimes
        # an age will exist without a corresponding time step.  Let's
        # correct that here
        if np.size(data_d[name]) > no_of_time_steps:
            data_d[name] = data_d[name][:-1]
        if verbose: print( now(), 'array= %(name)s, size=' % vars(), \
            np.size(data_d[name]) )

    # cumulative CPU time
    data_d['time_cumsum_per_step'] = np.cumsum( data_d['time_per_step'] )
    if verbose: print( now(), 'array= time_cumsum, size=' % vars(), \
        np.size(data_d['time_cumsum_per_step']) )


    data_d['time_cumsum_per_step_hours'] = data_d['time_cumsum_per_step'] \
                                             / 3600
    data_d['time_cumsum_per_step_days'] = data_d['time_cumsum_per_step'] \
                                             / (3600*24)

    return data_d

#=====================================================================
#=====================================================================
#=====================================================================

if __name__ == "__main__":

    main()

#====================================================================
#====================================================================
#====================================================================
