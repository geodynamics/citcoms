#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: TODO 
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This module holds functions for use with the Geodynamic Framework.'''
#=====================================================================
# TODO : put overview documentation here:
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
