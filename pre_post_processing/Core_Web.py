#!/usr/bin/env python
#=====================================================================
#                  Python Scripts for Data Assimilation 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2014
#                        ALL RIGHTS RESERVED
#=====================================================================
''' description here '''
#=====================================================================
#=====================================================================
import os, sys, subprocess
import numpy as np
import Core_Util

import urllib.parse
import urllib.request

#====================================================================
verbose = True
#====================================================================
#====================================================================
def test_ws():
    '''test the web services with a POST request'''

    # Get the sample json input 
    json_file = open('./geodynamic_framework_data/TEST.json', 'r')
    json_string = json_file.readline()
    json_file.close()

    # create the URL for a POST req
    url = 'http://gplates.gps.caltech.edu:8080/reconstruct_feature_collection/?'

    # set up the URL params
    values = { 
      'feature_collection':json_string,  
      'time':'0', 
      'output':'geojson', 
      'test': 'True',
    }
    if verbose : Core_Util.tree_print( values )

    data = urllib.parse.urlencode(values)
    data = data.encode('utf-8')
    req = urllib.request.Request(url, data)
    rsp = urllib.request.urlopen(req)

    content = rsp.read()
    print('content =', content)

#====================================================================
#====================================================================
def test( argv ):
    '''self test'''
    global verbose
    verbose = True 

    test_ws();
#====================================================================
#====================================================================
def usage():
    sys.exit()
#====================================================================
#====================================================================
if __name__ == "__main__":
    import Core_Web

    if len( sys.argv ) > 1:

        # process sys.arv as file names for testing 
        if sys.argv[1] == '-t':
            test( sys.argv )
            sys.exit(0)
    else:
        # print module documentation and exit
        help(Core_Web)

#====================================================================
#====================================================================
#====================================================================
