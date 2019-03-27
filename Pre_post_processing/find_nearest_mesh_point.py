#!/usr/bin/env python
#
#=====================================================================
#
#               Python Scripts for CitcomS Data Assimilation
#                  ---------------------------------
#
#                              Authors:
#                    Dan J. Bower, Michael Gurnis
#          (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#
#
#=====================================================================
#
#  Copyright 2006-2013, by the California Institute of Technology.
#
#  Last update: 2nd July 2013 by DJB
#=====================================================================
import Core_Citcom, Core_Util
import numpy as np
import operator, time
from Core_Util import now
from scipy import spatial
#=====================================================================
#=====================================================================
#=====================================================================
def main():

    # parameters
    nlon = 30
    nlat = 30

    # preliminaries
    master_d = Core_Citcom.get_all_pid_data( 'pid23039.cfg' )
    coor_by_cap = make_coordinate_files( master_d )

    # algorithm 1: brute force
    t0 = time.time()
    for nn in range(10):
        brute_force( master_d, coor_by_cap, nlon, nlat )
    t1 = time.time()
    total = t1-t0
    print( now(),' brute_force=', total )

    #t1 = timeit.timeit(stmt='brute_force_algorithm()', 
    #    setup='from __main__ import brute_force_algorithm')
    #print( t1 )

    # algorithm 2: kd tree
    # specific preliminaries
    coor_by_cap = Core_Util.flatten_nested_structure( coor_by_cap )
    coor_by_cap = np.array( coor_by_cap )
    tree = spatial.KDTree( coor_by_cap )
    #pts = np.array( [[0, 0],[1,2],[30,40],[56,56],[180,76],[240,-24],
    #    [270,-60],[37,5],[345,3],[356,-87]] )

    pts = np.array([30,30])
    t0 = time.time()
    print( tree.query( pts )[1] )
    t1 = time.time()
    total = t1-t0
    print( now(), 'kd_tree=', total )

#=====================================================================
#=====================================================================
#=====================================================================
def make_coordinate_files( master_d ):

    '''Make coordinate files'''

    coord_file = '/net/beno2/nobackup1/danb/input/mkhist/test/Coord/'
    pid_d = master_d['pid_d']
    coord_file += pid_d['datafile'] + '.coord.#'

    coor_by_cap =  Core_Citcom.read_citcom_surface_coor( pid_d,
                                                          coord_file )
    outname = 'coord.cap.#'
    coor_cap_names = Core_Citcom.write_cap_or_proc_list_to_files( pid_d,
                               outname, (coor_by_cap,), 'cap', False )

    return coor_by_cap

#=====================================================================
#=====================================================================
#=====================================================================
def brute_force( master_d, coor_by_cap, nlon, nlat ):

    pid_d = master_d['pid_d']
    nodex = pid_d['nodex']
    nodey = pid_d['nodey']
    nproc_surf = pid_d['nproc_surf']

    nearest_point = []
    for cc in range( nproc_surf ):
        ccoor = coor_by_cap[cc]
        nearest_point.append( cc )
        nearest_point[cc] = []
        for cline in ccoor:
            ( zlon, zlat ) = cline
            # compute distance between points
            zdist = Core_Util.get_distance( nlon, nlat, zlon, zlat )
            nearest_point[cc].append( zdist )

    # get index of nearest node to this coordinate
    fnearest_point = Core_Util.flatten_nested_structure( nearest_point )
    fnearest_coor = Core_Util.flatten_nested_structure( coor_by_cap )
    min_index, min_value = min(enumerate(fnearest_point), key=operator.itemgetter(1))
    (zzlon, zzlat) = fnearest_coor[min_index]

    print( now(), nlon, nlat, min_index, min_value, zzlon, zzlat )

    # find location of node in cap list
    cap_index = int( min_index / (nodex*nodey) )
    entry_index = min_index % (nodex*nodey)

    print( now(), cap_index, entry_index )

#=====================================================================
#=====================================================================
#=====================================================================

if __name__ == "__main__":

    main()

#=====================================================================
#=====================================================================
#=====================================================================
