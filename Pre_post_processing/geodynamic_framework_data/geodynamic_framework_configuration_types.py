#!/usr/bin/env python3.3
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: TODO 
#
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================

# This file holds the specialized type associations.
# 
# In general when the geodynamic modeling python framwork reads a configuration file,
# the variable and type assocications are automatically determined 
# by detailed parsing of the textual form of the value.
# 
# However, there are special cases where the exact type cannot be determined,
# (for example: variable=1 could be an int, or a float)
# and this file holds those special cases 
# 
# Please keep this list sorted alphabetically in each section
#
type_map = {

    # general forms for use in control .cfg files
    'time_spec' : 'string' , 
    'restart_ages' : 'string' , 

    # pid0000.cfg type files
    'bottbcval' : 'float' ,
    'botvbxval' : 'float' ,
    'botvbyval' : 'float' ,
    'buoyancy_ratio': 'float_list',
    'cdepv_ff' : 'float_list',
    'cp' : 'float' ,
    'density' : 'float' ,
    'dissipation_number' : 'float' ,
    'fi_max' : 'float' ,
    'fi_min' : 'float' ,
    'gravacc' : 'float' ,
    'gruneisen' : 'float' ,
    'Q0' : 'float' ,
    'radius' : 'float' ,
    'radius_inner' : 'float' ,
    'radius_outer' : 'float' ,
    'rayleigh' : 'float' ,
    'refvisc' : 'float' ,
    'thermdiff' : 'float' ,
    'thermexp' : 'float' ,
    'theta_max' : 'float' ,
    'theta_min' : 'float' ,
    'toptbcval' : 'float' ,
    'topvbxval' : 'float' ,
    'topvbyval' : 'float' ,
    'visc0' : 'float_list' ,
    'viscE' : 'float_list' ,
    'viscT' : 'float_list' ,
    'viscZ' : 'float_list' ,
    'visc_max' : 'float' ,
    'visc_min' : 'float' ,
    'walltime' : 'string' ,

    # make_history_for_age.py
    'plate_velocity' : 'float' ,
    'plate_velocity_theta' : 'float' ,
    'plate_velocity_phi' : 'float' ,
    'stencil_age' : 'float_list' ,
    'stencil_value' : 'int_list'
}
