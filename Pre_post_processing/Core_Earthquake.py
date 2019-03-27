#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mike Gurnis
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This module holds Earthquake realted functions for use with the Geodynamic Framework.'''
#=====================================================================
#=====================================================================
import datetime, os, pprint, re, subprocess, string, sys, traceback
import numpy as np
import random

# load the system defaults
sys.path.append( os.path.dirname(__file__) + '/geodynamic_framework_data/')
import geodynamic_framework_configuration_types

#=====================================================================
# Global variables
verbose = False 
#====================================================================
#====================================================================

import os, string, sys, math, time

import Core_Util
from Core_Util import now 
# TODO: add more imports as needed 
#import Core_GMT, GMT_Utilities, Mat_Utilities

import obspy
from obspy.fdsn import Client

# TODO: see this page for more info on URL in python
# https://docs.python.org/3.4/library/urllib.html
import urllib.parse
import urllib.request

#=====================================================================
#=====================================================================
def get_IRIS_WebServices_Catalog():
    '''Get the IRIS catalog and print the response'''

    client = Client('IRIS')
    starttime = obspy.core.utcdatetime.UTCDateTime('2014-01-01')
    endtime = obspy.core.utcdatetime.UTCDateTime('2014-05-25')

    catalog = client.get_events(starttime=starttime,endtime=endtime,minmagnitude=4,catalog='ISC') 
    print('catalog', catalog)

    #response = urllib.request.urlopen('http://service.iris.edu/fdsnws/event/1/query?starttime=2010-02-27T06:30:00&endtime=2011-04-01T06:30:00&catalog=GCMT&orderby=time&format=text&nodata=404')
    #print(response)
    
#=====================================================================
def get_CMT_Catalog(mode):
    ''' FIXME: ''' 

    if mode == 1:
        CMT_Catalog='/net/holmes/scratch2/gurnis/Earthquake_Catelogs/CMT_Catalog/jan76_dec10.ndk'
        cmt_simple='cmt_lon_lat_d_mb.xydm'

    CMT=open(CMT_Catalog)
    CMT_out=open(cmt_simple,'w')

    while 1:
        # Unravel the ndk format file
        line1=CMT.readline()
        line2=CMT.readline()
        line3=CMT.readline()
        line4=CMT.readline()
        line4=CMT.readline()
        if(line1):
            catalog=line1[0:3]
            date=line1[5:16]
            time=line1[16:25]
            lat=line1[27:33]
            lon=line1[34:42]
            depth=line1[42:47]
            mag=line1[48:55]
            location=line1[56:81]
            mb,ms=mag.split()
            CMT_out.write('%s %s %s %s\n' % (lon,lat,depth,mb))
        else:
            break

    CMT.close()
    CMT_out.close()

    return cmt_simple
#=====================================================================
def get_EHB_Catalog():
    EHB_Catalog='/net/holmes/scratch2/gurnis/Earthquake_Catelogs/ISC_Catalogs/EHB_1960_2008.dat'

    EHB=open(EHB_Catalog)
    ehb_simple='ehb_lon_lat_d_mb.xydm'
    EHB_out=open(ehb_simple,'w')

    while 1:
        # Unravel the ndk format file
        line=EHB.readline()
        if(line):
            event, author, date, time, lat, lon, depth, depfix, mag_author, type, mag = line.split(',')
            mb=float(mag)
            EHB_out.write('%s %s %s %g\n' % (lon,lat,depth,mb))
        else:
            break

    EHB.close()
    EHB_out.close()

    return ehb_simple
#=====================================================================
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this module'''

    text = '''#=====================================================================
# example config.cfg file for the Core_Earthquake
# ... 
# 
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
def test( argv ):
    '''geodynamic framework module self test'''
    global verbose
    verbose = True 
    print(now(), 'test: sys.argv = ', sys.argv )
    # run the tests 

    # read the defaults
    frame_d = Core_Util.parse_geodynamic_framework_defaults()

    # read the first command line argument as a .cfg file 
    #cfg_d = parse_configuration_file( sys.argv[1] )

    # TODO : comment in and out functions as needed 

    #get_IRIS_WebServices_Catalog()

    #get_CMT_Catalog(1)

    get_EHB_Catalog()

#=====================================================================
#=====================================================================
if __name__ == "__main__":
    import Core_Earthquake

    if len( sys.argv ) > 1:

        # make the example configuration file 
        if sys.argv[1] == '-e':
            make_example_config_file()
            sys.exit(0)

        # process sys.arv as file names for testing 
        test( sys.argv )
    else:
        # print module documentation and exit
        help(Core_Earthquake)

#=====================================================================
#=====================================================================
