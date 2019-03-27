#!/usr/bin/env python

import Core_GMT5
from Core_GMT5 import callgmt
import Core_Util
from plot_one import dataset_parameters
import numpy as np
import os, subprocess
from subprocess import Popen, PIPE

#====================================================================
#====================================================================
#====================================================================
def main():

    # THERMOCHEMICAL MODELS
    TC = 1
    #TC = 0
    model = 'mcm46'
    #model = 'mcm47'
    #model = 'mcm49'

    # THERMAL-ONLY MODELS
    #TC = 0
    #model = 'mcm47t'
    #model = 'mcm47u'
    #model = 'mcm47v'
    #model = 'M1'

    mcm_d = dataset_parameters()[model]

    # make track file
    track_filename = 'coordinates.xyp'
    opts_d = {}
    opts_d['D'] = 'g'
    opts_d['G'] = 0.5 # degrees

    # Africa
    opts_d['C'] = '-66.42/-57.67' # lon, lat, drake passage event (01/03/95)
    #opts_d['E'] = '34.35/4.96' # lon, lat (Tanzania station MBWE)
    opts_d['E'] = '71.11/36.40' # 2015-01-06 20:21:48.6 UTC, mb 4.4
    # Pacific
    # using data thief on 2009 He JGR cross-section
    #opts_d['C'] = '104/51'
    #opts_d['E'] = '267/-55'
    # when L=w only one coordinate is output
    #opts_d['L'] = 'w'
    #opts_d['Q'] = ''
    callgmt( 'project', '', opts_d, '>', track_filename )
    del opts_d['C']
    del opts_d['D']
    del opts_d['E']
    del opts_d['G']
    #del opts_d['Q']

    make_grid( mcm_d, model, track_filename, 'temp' )

    # thermochemical models only
    if TC:
        make_grid( mcm_d, model, track_filename, 'comp_nd' )

    return model

#====================================================================
#====================================================================
#====================================================================
def make_grid( mcm_d, model, track_filename, field ):

    pfx = field + '_'
    age_l = mcm_d['age_l']
    depth_l = mcm_d['depth_l']

    ps_l = []

    age_l = [2]

    for age in age_l:

        # reset dictionary
        opts_d = {}
        subprocess.call( ['rm', pfx + 'data.xyz'] )
        master_filename = pfx + 'data.xyz'
        master_file = open( master_filename, 'ab' )

        if field == 'temp':
            # store average temperature to compute best-fit
            # adiabat
            avg_temp_filename = 'average_temp_%(age)s.xy' % vars()
            avg_temp_file = open( avg_temp_filename, 'w' )

        for depth in depth_l:
            out_filename = pfx + 'data_%(depth)s_%(age)s.xyz' % vars()
            radius = 6371 - depth
            data_d = dataset_parameters( age, depth )[model]
            grid = data_d['field_d'][field]['grid_loc']
            opts_d['G'] = grid

            if field == 'temp':
                # find mean of grid to remove background for temperature
                mean_filename = pfx + 'data_%(depth)s_%(age)s_mean.nc' % vars()
                callgmt( 'grdmath', grid + ' MEAN ', '', '=', mean_filename )

                p1 = Popen( ['/opt/local/bin/gmt', 'grdinfo', mean_filename],
                    stdout=PIPE, stderr=PIPE )
                stdout, stderr = p1.communicate()
                # find mean value (z_min or z_max)
                stdout = stdout.decode().split('z_min')
                mean_val = float(stdout[1].split(' ')[1])
                print( 'mean_val=', mean_val )
                # remove adiabat instead!  (linear fit)
                # to mcm47u at 0 Ma
                #adi_val = 3.95949441E-01 + depth*6.34837862E-05

                avg_temp_file.write( '%(depth)s %(mean_val)s\n' % vars() )

            # track through grid
            callgmt( 'grdtrack', track_filename, opts_d, '>', out_filename )
            del opts_d['G']
            lon, lat, dist, val = np.loadtxt( out_filename, unpack=True )
            max_dist = np.max( dist )
            num = np.size( dist )
            rad = np.tile( np.array([radius]), num )
            # remove average from val
            if field == 'temp':
                val -= mean_val
            #    val -= adi_val # mean_val
            out_a = np.column_stack( (dist, rad, val) )
            np.savetxt( master_file, out_a )

        master_file.close()
        if field == 'temp':
            avg_temp_file.close()

        # make grid file
        block_filename = pfx + 'blockmedian.xyz'
        opts_d['I'] = '1/80' # 2/100 OK was 0.5/50
        opts_d['R'] = '0/%(max_dist)s/3504/6371' % vars()
        callgmt( 'blockmedian', master_filename, opts_d, '>', block_filename )
        del opts_d['I']

        opts_d['G'] = pfx + 'cross_section_%(age)s.nc' % vars()
        opts_d['I'] = '0.25/20'  # 1/80 OK
        #opts_d['L'] = 'l0 -Lu1'
        opts_d['T'] = '0.3'
        callgmt( 'surface', block_filename, opts_d, '', '' )
        del opts_d['G']
        del opts_d['I']
        del opts_d['T']

        ps = quick_plot( model, age )
        ps_l.append( ps )

    Core_Util.make_pdf_from_ps_list( ps_l, model+'.pdf' )

#====================================================================
#====================================================================
#====================================================================
def quick_plot( model, age ):

    ps = model + '_cross_section_%(age)s.ps' % vars()
    opts_d = Core_GMT5.start_postscript( ps )

    opts_d['B'] = 'a20/1000::WeSn'
    opts_d['C'] = '/Users/dan/Documents/research/plotting/cpt/dtemp.cpt'
    #opts_d['C'] = '/Users/dan/Documents/research/plotting/cpt/polar_0_1_10_Z.cpt'
    opts_d['J'] = 'Pa7/72.5z'
    opts_d['R'] = '0/145/3505/6371'
    opts_d['X'] = 'a1'
    opts_d['Y'] = 'a5'

    callgmt( 'grdimage', 'temp_cross_section_%(age)s.nc' % vars(), opts_d, '>>', ps )

    del opts_d['B']

    comp_file = 'comp_nd_cross_section_%(age)s.nc' % vars()
    if os.path.isfile( comp_file ):
        opts_d['C'] = '+0.7'
        opts_d['W'] = '2,green'
        #opts_d['X'] = 'a6'
        callgmt( 'grdcontour', comp_file, opts_d, '>>', ps )

    Core_GMT5.end_postscript( ps )

    return ps

#====================================================================
#====================================================================
#====================================================================

if __name__ == "__main__":

    main()
