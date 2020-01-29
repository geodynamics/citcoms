#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan Bower, Nicolas Flament 
#
#                  ---------------------------------
#             (c) California Institute of Technology 2013, 2014
#                        ALL RIGHTS RESERVED
#=====================================================================
'''restart_citcoms.py is the top-level script to create an populate a 
set of citcoms model run directories suitable for restart processing.'''
#=====================================================================
#=====================================================================

# Import Python and standard modules 
import sys, string, os, copy
import numpy as np

# Import geodynamic framework Core modules
import Core_Citcom
import Core_Util
from Core_Util import now

Core_Util.verbose = False
Core_Util.verbose = True
Core_Citcom.verbose = False
Core_Citcom.verbose = True

#=====================================================================
#=====================================================================
# Global Variables for this script
verbose = False
verbose = True
#=====================================================================
#=====================================================================
def usage():
    '''print usage message and exit'''
    print('''restart_citcoms.py [-e] configuration_file.cfg 

    restart_citcoms.py is a tool to create a set of Citcom input .cfg files 
    for a set of restart runs based upon an existing prior master Citcom run.

    It makes adjusted copies of the master run's pidXXX.cfg file to become 
    a new set of input .cfg files for the new restart runs.

    It uses default parameters from the Core_Citcom.py module, 
    and from its own input .cfg file, to modify the master run .cfg 
    to create new restart input .cfg files.

    For 'dynamic_topogrphy' restart runs this script also makes modified 
    copies of the temperature files for input to the restart runs.

    Run this script under the same directory of the citcomS master run cfg and pid directory.
    Please run this script with '-e' and read the sample input config file for more details.
''')

    sys.exit(0)
#=====================================================================
#=====================================================================
def main() :
    '''This is the main function for restart_citcoms.py'''
    print( now(), 'restart_citcoms.py: START')

    # get the control .cfg file as a dictionary 
    control_d = Core_Util.parse_configuration_file(sys.argv[1])

    # parse master run input pid file 
    master_run_cfg = control_d['master_run_cfg']
    master_run_cfg_d = Core_Util.parse_configuration_file(master_run_cfg)

    # parse master run output pid file 
    master_run_pid = control_d['master_run_pid']

    # get the master dictionary and define aliases
    master_run_d = Core_Citcom.get_all_pid_data( master_run_pid )
    master_run_d['control_d'] = control_d
    master_run_pid_d = master_run_d['pid_d']

    # report details of input data 
    if verbose: 
        print( now(), 'restart_citcoms: control_d = ')
        Core_Util.tree_print(control_d)
        print( now(), 'restart_citcoms: master_run_cfg_d = ')
        Core_Util.tree_print(master_run_cfg_d)
        print( now(), 'restart_citcoms: master_run_pid_d = ')
        Core_Util.tree_print(master_run_pid_d)

    # SAVE, might need later ... ?
    # copy of the geo frame defaults
    #geoframe_d = master_run_d['geoframe_d']

    # process the control entry to get a list of ages
    time_spec_d = Core_Citcom.get_time_spec_dictionary( control_d['restart_ages'], master_run_d['time_d'] )

    print( now(), 'restart_citcoms: time_spec_d =')
    Core_Util.tree_print(time_spec_d)

    # Get the restart type and local copy of the restart parameter replacement dictionary
    rs_replace_d = {}
    rs_type = control_d['restart_type']
    if rs_type == 'dynamic_topography' :
       rs_replace_d = Core_Citcom.dynamic_topography_restart_params
    elif rs_type == 'total_topography' :
       rs_replace_d = Core_Citcom.total_topography_restart_params
    else: 
        print( now(), 'restart_citcoms: ERROR: unknown value for restart_type.')
        print( now(), 'Valid values are "dynamic_topography", or "total_topography"')
        sys.exit(-1)

    # Now update rs_replace_d values directly from those set in control_d 
    for p in sorted(control_d) :
        if p.startswith('CitcomS.') :
            rs_replace_d[p] = control_d[p]

    # Show the final rs_replace_d that will pass to the input creation function
    if verbose: 
        print( now(), 'restart_citcoms: rs_replace_d = ')
        Core_Util.tree_print(rs_replace_d)

    # Set placeholders for the directory and file structre  and names
    rs_dir_prefix = 'restart_' + rs_type
    rs_inp_cfg_suffix = ''

    rs_structure = control_d['restart_structure']
    if rs_structure == 'all-in-one' :
        # create the all-in-one restart directory from section name
        Core_Util.make_dir( rs_dir_prefix ) 
    
    # Now it's time to Loop over restart ages and create restart files for that age
    for a in time_spec_d['age_Ma'] :

        # determine what time steps are available for this age 
        # NOTE: 'temp' is requried to set which output files to check 
        found_d = Core_Citcom.find_available_timestep_from_age( master_run_d, 'temp', a )

        timestep = found_d['found_timestep'] 
 
        # convert the found age to an int
        age = int(np.around( found_d['found_age'] ) )

        print( now(), '--------------------------------------------------------------------------------------------')
        print( now(), 'Creating files for restart run at age:', age, '(', str(a), 'Ma; timestep = ', timestep, ')' )
        print( now(), '--------------------------------------------------------------------------------------------')

        # Set the name of the restart directory
        rs_dir = ''
        if rs_structure == 'separate':
            # create restart directory from section name
            rs_dir = rs_dir_prefix + '_' + str(age) + 'Ma' 
            Core_Util.make_dir( rs_dir ) 
        else: 
            # this is an all-in-on case
            rs_dir = rs_dir_prefix

        # update the new restart input cfg file name suffix
        rs_inp_cfg_suffix = rs_type + '_' + str(age) + 'Ma'

        # create a new set of initial conditions for the restart run, 
        # and set file name patterns in control_d
        if rs_type == 'dynamic_topography' : 
            create_no_lith_temp( 
                control_d, master_run_d, rs_replace_d, rs_dir, rs_inp_cfg_suffix, age, timestep)

        # else, no need to adjust files for 'total_topography' runs
 
        # create new run input .cfg for this restart run
        restart_run_cfg = {}
        restart_run_cfg = create_restart_run_cfg(
            control_d, master_run_cfg_d, rs_replace_d, rs_dir, rs_inp_cfg_suffix, age, timestep)

    # End of loop over restart runs 

    # Close up shop
    sys.exit(0)
#=====================================================================
#=====================================================================
def create_restart_run_cfg(control_d, master_run_cfg_d, rs_replace_d, rs_dir, rs_inp_cfg_suffix, age, timestep) :
    '''return a copy of master_run_cfg_dict with adjustments as set in Core_Citcom and control_d'''

    # get a copy of the restart type
    rs_type = control_d['restart_type']

    # make a local copy of the master run input cfg 
    restart_run_cfg_d = {}
    restart_run_cfg_d = copy.deepcopy(master_run_cfg_d)

    # Make changes to local restart_run_cfg_d based upon control_d and rs_replace_d

    # Loop over all the params in rs_replace_d ; these are the values to update for restart
    for p in sorted(rs_replace_d) :

        # Get the replacement value 
        val = rs_replace_d[p] 
 
        # Get the subsection name and param name
        strings = p.split('.')
        section_name = '.'.join( strings[0:-1] )
        param_name = strings[-1]
        if verbose: 
            print( now(), 'create_restart_run_cfg: param = ', p, '; section_name = ', section_name, '; param_name =', param_name, '; val =', val)

        # make sure this section name exists in the cfg dictionary, just in case
        if not section_name in restart_run_cfg_d:
            print( now(), 'WARNING: section_name', section_name, ' not found in orginal cfg.  Adding entry to new restart cfg.')
            restart_run_cfg_d['_SECTIONS_'].append( section_name )
            restart_run_cfg_d[section_name] = {}
            restart_run_cfg_d[section_name]['_SECTION_NAME_'] = section_name
            restart_run_cfg_d[section_name][param_name] = None

        # Check if this value is to be deleted 
        if val == 'DELETE' : 
            if param_name in restart_run_cfg_d[section_name] :
                del restart_run_cfg_d[section_name][param_name]
            
        # check if this parameter is to be commentd out in the new cfg
        elif val == 'COMMENT' :
 
            # double check this param has a orginal value in the restart_run_cfg_d
            if not param_name in restart_run_cfg_d[section_name]:
                continue # the parm name was not in the master run cfg ; skip it 

            # else get master run cfg value 
            master_run_val = restart_run_cfg_d[section_name][param_name]
            # remove original param name
            del restart_run_cfg_d[section_name][param_name]
            # add new value and commented out 
            restart_run_cfg_d[section_name][ '# ' + param_name ] = master_run_val

        elif val == 'RS_TIMESTEP':
            restart_run_cfg_d[section_name][param_name] = timestep
        elif val == 'RS_TIMESTEP+2':
            restart_run_cfg_d[section_name][param_name] = timestep + 2


        else: 
           # this is a regular value, update restart_run_cfg_d
           restart_run_cfg_d[section_name][param_name] = val

        # Double check if this param is set in the control cfg 
        if section_name + '.' + param_name in control_d :
            val = control_d[section_name + '.' + param_name]
            restart_run_cfg_d[section_name][param_name] = val

    # Now set some specific values based upon restart type, age and timestep 

    # Get a copy of the master run datafile value
    if 'CitcomS.solver' in restart_run_cfg_d and all(x in restart_run_cfg_d['CitcomS.solver'] for x in ['datafile', 'datadir']):
        master_run_datafile = restart_run_cfg_d['CitcomS.solver']['datafile']
        master_run_datadir = restart_run_cfg_d['CitcomS.solver']['datadir']
    elif all(x in restart_run_cfg_d for x in ['datafile', 'datadir']):
        master_run_datafile = restart_run_cfg_d['datafile']
        master_run_datadir = restart_run_cfg_d['datadir']
    else:
        sys.exit('unable to find data files!!')
    # Set the new values for datafile, datafile_old, datadir, datadir_old 

    # FIXME: are the new values for these 4 params ^ ^ ^ ^  
    # set correctly in the two cases below?

    if rs_type == 'total_topography' :

        restart_run_cfg_d['CitcomS.solver']['datafile'] = master_run_datafile
        restart_run_cfg_d['CitcomS.solver']['datadir'] =  'Age' + str(age) + 'Ma'

        # FIXME: DJB, TY, and NF to double check this change from above:
        if os.path.exists( master_run_cfg_d['datadir']+ '/0/') :
            restart_run_cfg_d['CitcomS.solver']['datafile_old'] = master_run_datafile
            restart_run_cfg_d['CitcomS.solver']['datadir_old'] =  os.path.normpath(os.path.join('../',master_run_cfg_d['datadir']))+'/%RANK'
        elif os.path.exists( master_run_cfg_d['datadir']+ '/' ) :
            restart_run_cfg_d['CitcomS.solver']['datafile_old'] = master_run_datafile
            restart_run_cfg_d['CitcomS.solver']['datadir_old'] =  os.path.normpath(os.path.join('../',master_run_cfg_d['datadir']))
        else :
            #restart_run_cfg_d['CitcomS.solver']['datafile_old'] = master_run_datafile 
            #restart_run_cfg_d['CitcomS.solver']['datadir_old'] = '../data/%RANK'
            restart_run_cfg_d['CitcomS.solver']['datafile_old'] = master_run_datafile
            restart_run_cfg_d['CitcomS.solver']['datadir_old'] =  '../data/%RANK'

    elif rs_type == 'dynamic_topography' :

        restart_run_cfg_d['CitcomS.solver']['datafile'] = master_run_datafile 
        restart_run_cfg_d['CitcomS.solver']['datadir'] = 'Age' + str(age) + 'Ma' 

        # FIXME: DJB, TY, and NF to double check this change:
        # NOTE: these two values are derrived in create_no_lith_temp()
        #restart_run_cfg_d['CitcomS.solver']['datafile_old'] = master_run_datafile
        #restart_run_cfg_d['CitcomS.solver']['datadir_old'] = '../%RANK'
        restart_run_cfg_d['CitcomS.solver']['datafile_old'] = control_d['rs_datafile']
        restart_run_cfg_d['CitcomS.solver']['datadir_old'] = os.path.normpath(control_d['rs_datadir'])

        restart_run_cfg_d['CitcomS.solver.param']['start_age'] = str(age)
    else:
        print(now(), 'ERROR: unknown restart type.  Value must be either "dynamic_topography" or "total_topography"')

    # coor_file needs special tackle. maybe can move into the loop above?
    if not os.path.isabs(restart_run_cfg_d['coor_file']):
        tmp1=os.path.normpath(os.path.join('../',restart_run_cfg_d['coor_file']))
        restart_run_cfg_d['coor_file']=tmp1

    # Write out the new input cfg dictionary
    cfg_name = rs_dir + '/' + master_run_datafile + '_' + rs_inp_cfg_suffix + '.cfg'
    Core_Util.write_cfg_dictionary( restart_run_cfg_d, cfg_name, False) 


    # And return it 
    return restart_run_cfg_d
#=====================================================================
#=====================================================================
def create_no_lith_temp(control_d, master_run_d, rs_replace_d, rs_dir, rs_inp_cfg_suffix, age, timestep):
    '''read master run velo files and modify the temperature using z>some_node '''
# (6) Read in velo file from master run for closest age (use read_proc_files_to_cap_list() )
# (7) Modify the temperature using z>some_node to set temperatures to background for models
#  without the lithosphere
# (8) write out `new' IC files using write_cap_or_proc_list_to_files()

    lithosphere_depth_DT = control_d['lithosphere_depth_DT'] 
    lithosphere_temperature_DT = control_d['lithosphere_temperature_DT'] 

    # Get nodez from depth
    znode = Core_Citcom.get_znode_from_depth( master_run_d['coor_d'], lithosphere_depth_DT )
    print (now(), 'create_no_lith_temp: lithosphere_depth_DT = ', lithosphere_depth_DT, '; znode=', znode)

    # choose the field to process
    field_name = 'temp'

    # get params for the run
    pid_d = master_run_d['pid_d']
    datafile = pid_d['datafile']

    # get the data file name specifics for this field 
    file_name_component = Core_Citcom.field_to_file_map[field_name]['file']
    print( now(), 'create_no_lith_temp: file_name_component = ', file_name_component )

    # process data from Citcoms 
    if os.path.exists( master_run_d['pid_d']['datadir']+ '/0/') :
        file_format = master_run_d['pid_d']['datadir']+ '/#/' + master_run_d['pid_d']['datafile'] + '.' + file_name_component + '.#.' + str(timestep)
    elif os.path.exists( master_run_d['pid_d']['datadir']+ '/' ) :
        file_format = master_run_d['pid_d']['datadir']+ '/' + master_run_d['pid_d']['datafile'] + '.' + file_name_component + '.#.' + str(timestep)
    elif os.path.exists( master_run_d['pid_d']['datadir'].replace( '%RANK', '0' )) :
        file_format = master_run_d['pid_d']['datadir'].replace( '%RANK', '#' )+ '/' + master_run_d['pid_d']['datafile'] + '.' + file_name_component + '.#.' + str(timestep)
    else :
        file_format = 'data/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)
    print( now(), 'create_no_lith_temp: create_no_lith_temp: file_format = ', file_format )

    # read data by proc, e.g., velo, visc, comp_nd, surf, botm 
    data_by_cap = Core_Citcom.read_proc_files_to_cap_list( master_run_d['pid_d'], file_format, field_name)

    # find index of all nodes in a cap that have znode > requested_znode
    # first, make array of znode number for a cap
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    nodez = pid_d['nodez']

    # znodes for one cap (same for every cap)
    znode_array = np.tile( range(nodez), nodex*nodey )

    # this gives  a mask of all the znodes that we need to correct the temperature for
    mask = np.where( znode_array > znode, True, False )

    # loop over all cap lists
    for nn, cap_list in enumerate(data_by_cap):
        print(now(), 'create_no_lith_temp: working on cap number', nn )
        # convert to numpy array
        cap_array = np.array(cap_list)
        # swap in new temperature values for lithosphere
        # temperature is fourth column
        np.place( cap_array[:,3], mask, lithosphere_temperature_DT )
        # update master list of data with corrected list
        data_by_cap[nn] = cap_array.tolist()

    # check values have been updated
    #if verbose: print( now(), 'create_no_lith_temp: spot check: data_by_cap[0][0:nodez]', data_by_cap[0][0:nodez])

    # map the data from cap lists to processor lists
    out_data_by_proc = Core_Citcom.get_proc_list_from_cap_list( master_run_d['pid_d'], data_by_cap )

    # set up output info 
    rs_datafile = datafile + '_restart_' + str(int(np.around(age))) + 'Ma'

    ic_dir = rs_dir + '/ic_dir'
    Core_Util.make_dir( ic_dir )
    out_name = ic_dir + '/' + rs_datafile + '.velo.#.' + str(timestep)
    print( now(), 'create_no_lith_temp: out_name =', out_name )

    # now write out data to processor files (with header, necessary for restart)
    Core_Citcom.write_cap_or_proc_list_to_files( 
        master_run_d['pid_d'], out_name, (out_data_by_proc,), 'proc', True )

    # Update control_d with file name patterns 
    control_d['rs_datafile'] = rs_datafile
    control_d['rs_datadir'] = './ic_dir/'

    return
#=====================================================================
#=====================================================================
#=====================================================================
# SAVE these NOTES during development
# (3) parse pid file from CitcomS master run (needed for general node and model set-up parameters)
# (4) read in CitcomS master run .cfg to use as template
# (5) modify master run .cfg for EACH restart model by updating, e.g.
# steps
# name
# walltime
# monitoringFrequency
# datafile
# datadir_old   *** must point to `new' initial condition files
# datafile_old   *** must point to `new' initial condition files
# start_age   *** will be closest age to model output
# solution_cycles_init   *** suffix for new ic file (usually an age)
# output_optional   *** ensure that surf is output
# modify topvbc

# (6) Read in velo file from master run for closest age (use read_proc_files_to_cap_list() )
#
# (7) Modify the temperature using z>some_node to set temperatures to background for models
#  without the lithosphere
#
# (8) write out `new' IC files using write_cap_or_proc_list_to_files()
#
# (9) also need to duplicate comp_nd file for each desired time step
# so the composition field can be used for restart as well (since this
# encodes the compositional buoyancy of the continents, deep Earth
# piles etc.)
#
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# This is an example config.cfg file for restart_citcoms.py
# 
# Set the name of the master run model input RUN.cfg and output PID.cfg files
master_run_cfg = mcm48.cfg
master_run_pid = pid32395.cfg


# Set a list of restart ages in Ma 
#
# Use a start/stop/step trio
#restart_ages = 0Ma/250Ma/50Ma
#
# or use an explicit list of one or more individual ages:
#restart_ages = 29Ma
restart_ages = 29Ma, 80Ma, 128Ma


# Set type of restart runs to create: choose one option:
#
# 'total_topography' cooresponds to 'checkpoint-template' in previous workflows
restart_type = total_topography 

# 'dynamic_topogrphy' cooresponts to 'nolith-template' in previous worklfows
#restart_type = dynamic_topography 
#
# dynamic_topogrphy restarts also require these two values to be set:
#
# depth to remove lithosphere, in km:
lithosphere_depth_DT = 350.0
#
# non-dimensional temperature value to replace in new velo files
lithosphere_temperature_DT = 0.5 


# Set the directory and file structure: choose one option:
#
# 'all-in-one' will create all new restart input .cfg and modified velo files 
# in a single directory with the prefix 'restart_' and suffix by restart_type.
#restart_structure = all-in-one 
#
# 'separate' will create separate directories for each new restart age,
# using the prefix 'restart_AGE' and suffix by restart type.
restart_structure = separate


# Restart input .cfg files will have most parameters set according to restart_type.
# (set in Core_Citcoms.py in the dynamic_topography_restart_params 
# and total_topography_restart_params dictionaries )
#
# User specific replacement values can appear in this file.
# Prefix parameters with the correct [CitcomS.*] section name.
# For example:

CitcomS.solver.bc.topvbc = 0
CitcomS.solver.output.output_ll_max = 80
CitcomS.solver.tsolver.finetunedt = 0.01

#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
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
#=====================================================================
#=====================================================================
