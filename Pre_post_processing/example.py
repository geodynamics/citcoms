#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan J. Bower
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                  ---------------------------------
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================
# example.py
#=====================================================================
# This script is the minimal example for the geodynamic framework.
#
# It shows the framework style guide for coding conventions,
# script invocation, funcation calls, usage messages, 
# creating example confiuguration file, 
# and other programming traits common to the project.
# 
# New scripts created for this framework should use this script as 
# a kind of template for script flow and organization.
# 
# Please see the usage() function below for more information.
#=====================================================================
#=====================================================================
import sys, string, os
import numpy as np
#=====================================================================
import Core_Citcom
import Core_GMT
import Core_Util
from Core_Util import now
#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    print('''usage: example.py [-e] configuration_file.cfg

Options and arguments:
  
-e : if the optional -e argument is given this script will print to standard out an example configuration control file.
   The parameter values in the example config.cfg file may need to be edited or commented out depending on intended use.

'configuration_file.cfg' : is a geodynamic framework formatted control file, with at least these entries: 

    string = a string to print 

See the example config.cfg file for more info.
''')
    sys.exit()
#=====================================================================
# ... more functions go here ... 
#=====================================================================
def main():
    '''This is the main workflow of the script'''

    # report the start time and the name of the script
    print( now(), 'example.py')

    # a way to show what version of python is being used:
    print( now(), 'sys.version_info =', str(sys.version_info) )

    print("os.path.dirname(__file__) =" , os.path.dirname(__file__) )


    # Get the configuration control file as a dictionary
    control_d = Core_Util.parse_configuration_file( sys.argv[1] )

    # Get the string 
    string = control_d['string']

    # print the string 
    print( now(), 'string =', string )

    sys.exit()

#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# config.cfg file for the example.py script
# 
# This is the minimal example.
string = hello world 
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
