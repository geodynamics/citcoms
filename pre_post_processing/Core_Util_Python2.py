#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This module holds general purpose functions for use with the Geodynamic Framework.'''

#=====================================================================
#=====================================================================
import datetime, os, pprint, re, subprocess, string, sys, traceback, math, copy
import numpy as np
import random
import bisect

# load the system defaults
sys.path.append( os.path.dirname(__file__) + '/geodynamic_framework_data/')
import geodynamic_framework_configuration_types

#=====================================================================
# Global variables

# turn verbose off by default; client code can call Core_Util_Python2.verbose = True
verbose = False 

# the character uses in GMT style .xy files to indicate a header line
GMT_HEADER_CHAR = '>'

# common place to hold the earth's radius in km
earth_radius = 6371.0

#====================================================================
#====================================================================
def now():
    '''Redefine now() with short format yyyy-mm-dd hh:mm:ss'''
    return str(datetime.datetime.now())[11:19]
#=====================================================================
#=====================================================================
def tree_print(arg):
    '''print the arg in tree form'''
    print(now())
    pprint.PrettyPrinter(indent=2).pprint(arg)
#=====================================================================
#=====================================================================
def parse_general_key_equal_value_linetype_file(filename):
    '''Parse a general 'key=value', one item per line, type file and return the data as a dictionary.
       NOTE: No subsections are allowed.  Use parse_configuration_file() if subsections are requried.
'''
    global verbose 

    # the dictionary to return
    ret_dict = {}

    # loop over lines in the file
    infile = open(filename,'r')
    for line in infile:
        if verbose: print(now(), "parse_general_key_equal_value_linetype_file: line ='", line, "'")

        # skip past comments and blank lines
        if line.startswith('#'):
            continue # to next line
        if line.rstrip() == '':
            continue # to next line

        # get and split the data
        key, val = line.split('=')
        key = key.strip() # clean up white space
        key = key.rstrip() # clean up white space
        val = val.strip() # clean up white space
        val = val.rstrip() # clean up white space
        
        if verbose: print(now(), "parse_general_key_equal_value_linetype_file: key=", key, '; val=', val)

        # add this entry to the dict
        ret_dict[key]=val
    # end of loop over lines

    return ret_dict
#=====================================================================
#=====================================================================
def parse_geodynamic_framework_defaults():
    '''Read the geodynamic_framework defaults file from of these locations:
            local: "./geodynamic_framework_defaults.conf"
            user home dir: "~/geodynamic_framework_defaults.conf"
            system defult: "/path/to/system"
       and return the data as a dictionary.

       If the local and user home files are missing, 
       then copy the system file to the current working directory.
'''
    # Alert the user as to which defaults file is being used:
    print( now(), 'Core_Util_Python2.parse_geodynamic_framework_defaults():' )
    # check for local framework defaults 
    file = './geodynamic_framework_defaults.conf'
    if os.path.exists(file):
        print(now(), 'Using the local current working directory Geodynamic Framework defaults file:\n', file)
        return parse_general_key_equal_value_linetype_file( file )

    # check for user's default file 
    file = os.path.expanduser('~/geodynamic_framework_defaults.conf')
    if os.path.exists(file):
        print(now(), 'Using the home directory Geodynamic Framework defaults file:\n', file)
        return parse_general_key_equal_value_linetype_file( file )

    # local and user files are missing, parse the default system file:

    # check for system default file
    file = os.path.abspath( os.path.dirname(__file__) ) + "/geodynamic_framework_data/geodynamic_framework_defaults.txt"
    if os.path.exists(file):
        print(now(), 'Using the System Geodynamic Framework defaults file:\n', file)

        # copy the system file to the cwd
        print(now(), 'Copying the System file to the current working directory as: "geodynamic_framework_defaults.conf"')
        cmd = 'cp ' + file + ' ' + 'geodynamic_framework_defaults.conf'
        print( now(), cmd )
        subprocess.call( cmd, shell=True )

        return parse_general_key_equal_value_linetype_file( file )

    # halt the script if we cannot find a valid file 
    print(now(), 'Cannot fine a valid geodynamic_framework defaults file.')
    print(now(), 'Please update from SVN to get /geodynamic_framework_data/geodynamic_framework_defaults.conf')
    sys.exit(1)
#=====================================================================
#=====================================================================
    return spec_d

#=====================================================================
#=====================================================================
#=====================================================================
def test( argv ):
    '''geodynamic framework self test'''

    global verbose
    verbose = True

    print(now(), 'Core_Util_Python2.test(): sys.argv = ', sys.argv )

    # read the defaults
    frame_d = parse_geodynamic_framework_defaults()

#=====================================================================
#=====================================================================
#=====================================================================
def test_function( args ):
    '''test specific functions'''
    global verbose 
    verbose = True 

    # NOTE: comment or uncomment the specfic functions to test here:

    file = args[2]
    print( now(), 'test_function: file =', file)

#=====================================================================
#=====================================================================
#=====================================================================
if __name__ == "__main__":
    import Core_Util_Python2

    if len( sys.argv ) > 1:

        # make the example configuration file 
        #if sys.argv[1] == '-e':
        #    make_example_config_file()
        #    sys.exit(0)

        # run a specific test on sys.argv
        if sys.argv[1] == '-t':
            test_function( sys.argv )
            sys.exit(0)

        # process sys.arv as file names for testing 
        test( sys.argv )
    else:
        # print module documentation and exit
        help(Core_Util_Python2)

#=====================================================================
#=====================================================================
