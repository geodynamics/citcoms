#!/usr/bin/env python3.3
#=====================================================================
#               Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This is a test script to show a simple example of using the geodynamic framework modules'''
#=====================================================================
#=====================================================================
import os, sys, datetime, pprint, subprocess

import Core_Util
import Core_Citcom

if not len(sys.argv) == 2:
    print('Run this script like this: ./test_script.py sample.cfg')
    sys.exit(-1)

# create an empty dictionary to hold all the main data
master_d = {}

# get the main data 
master_d = Core_Citcom.get_master_data( sys.argv[1] )

# show the dict
print('\n', Core_Util.now(), 'master_d = ')
Core_Util.tree_print(master_d)

# do something with the data ... 
print()
print('the pid file has nx =', master_d['pid']['nx'], 'nodes')
print('the coor file has', len( master_d['coor']['depth'] ), 'levels')

#=====================================================================
