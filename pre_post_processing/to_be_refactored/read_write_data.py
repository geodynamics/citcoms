#!/usr/bin/env python
import Core_Citcom as cc
import Core_Util as cu
import numpy as np

# FYI Mark Turner to see how to read and write processor and cap data
# updated to amend temperature field for dynamic topography (DT)
# restart
# written by Dan J. Bower, 11/21/13

# This takes a bit of time to run since it is crunching a lot of data
# see generated output here:
#     /home/danb/mark/read_write

# read CitcomS data
inp_file = '/net/beno2/nobackup1/danb/regional/ram/gam01/data/#/gam01.velo.#.0'
pid_file = '/net/beno2/nobackup1/danb/regional/ram/gam01/pid24611.cfg'

# get dictionaries etc
pid_d = cu.parse_configuration_file( pid_file )
pid_d.update( cc.derive_extra_citcom_parameters( pid_d ) ) 

# read in data to a cap list from processor files
data_by_cap = cc.read_proc_files_to_cap_list( pid_d, inp_file, 'temp' )

##########################################
##### UPDATE LITHOSPHERE TEMPERATURE #####
##########################################

# user specified in cfg
# where dt stands for 'dynamic topography'
lithosphere_depth_DT = 300.0 # km
lithosphere_temperature_DT = 0.5 # non-dimensional

# find out which node lithosphere_depth_dt corresponds to by searching in depth list
# TODO
# let's say node 52 for arguments sake
# note that nodes run from 0 to nodez-1!
znode = 52

# find index of all nodes in a cap that have znode>52
# first, make array of znode number for a cap

# Mark - this is why you still need the pid_d dictionary
nodex = pid_d['nodex']
nodey = pid_d['nodey']
nodez = pid_d['nodez']

# znodes for one cap (same for every cap)
znode_array = np.tile( range(nodez), nodex*nodey )
# this gives  a mask of all the znodes that we need to correct the temperature for
mask = np.where( znode_array > znode, True, False )


# loop over all cap lists
for nn, cap_list in enumerate(data_by_cap):
    print( 'working on cap number', nn )
    # convert to numpy array
    cap_array = np.array(cap_list)
    # swap in new temperature values for lithosphere
    # temperature is fourth column
    np.place( cap_array[:,3], mask, lithosphere_temperature_DT )
    # update master list of data with corrected list
    data_by_cap[nn] = cap_array.tolist()

# check values have been updated
print( data_by_cap[0][0:nodez] )

# let's map the data from cap lists to processor lists
data_by_proc = cc.get_proc_list_from_cap_list( pid_d, data_by_cap )

# now write out data to processor files (with header, necessary for restart)
# this suffix identifier (integer) could be a reference, a time step, or an age
# we'll need to decide what is the best to use
restart_id = 1
file_template = 'restart_DT.velo.#.%(restart_id)s' % vars()
cc.write_cap_or_proc_list_to_files( pid_d, file_template, (data_by_proc,), 'proc', True )


##########################################
##### END OF LITHOSPHERE TEMPERATURE #####
##########################################


# type should be a list (of lists)
#print( type(data_by_cap) )
# number of entries in cap 0 (should be nodex*nodey*nodez)
#print( len(data_by_cap[0]) )
# number of entries in cap 1 (should be nodex*nodey*nodez)
#print( len(data_by_cap[1]) )

# write out data to cap files
#cc.write_cap_or_proc_list_to_files( pid_d, 'out0.cap.#', (data_by_cap,), 'cap', False )

# let's map the data from cap lists to processor lists
#data_by_proc = cc.get_proc_list_from_cap_list( pid_d, data_by_cap )

# note that all lines are stored from the velo files (specifying 'temp' is only used to remove the header, and not extract the temp column).  If you want to extract data you'll need to loop over the data and split lines etc.


# now write out data to processor files (without header)
#cc.write_cap_or_proc_list_to_files( pid_d, 'out0.proc_no_head.#', (data_by_proc,), 'proc', False )

# now write out data to processor files (with header, necessary for restart)
#cc.write_cap_or_proc_list_to_files( pid_d, 'out0.proc_head.#', (data_by_proc,), 'proc', True )
