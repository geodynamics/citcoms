#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Laura Alisic, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This script uses the Geodynamic Framework to post-process Rhea model data.'''
#=====================================================================
#=====================================================================
# plot_Rhea_xsection_compression.py
#
# Processes paraview *.csv output into files readable
# by GMT, and plots xsections with desired quantities:
#   * compression from Rhea
#   * interpolated compression from CMTs
#   * slab outline in light grey
#
# Usage: 
# plot_Rhea_xsection_compression.py <run_name>
# (from within <rundir>/xsections_csv)
#
# Example for the reference model:
# plot_Rhea_xsection_compression.py run104
#
# Created: 06/20/2011
# Last modified: 07/04/2011
# Laura Alisic, Caltech
##################################################

# Import Python and standard modules 
import sys, string, os
import numpy as np
from  math import *

# Import geodynamic framework Core modules
import Core_Util
from Core_Util import now
Core_Util.verbose = False

import Core_Rhea
Core_Rhea.verbose = True

#=====================================================================
#=====================================================================
# Global Variables
verbose = True

# FIXME : replace with numpy values
r2d = 180.0/pi
d2r = 1.0/r2d

# FIXME : replace with values set in a Core_ Module ? 
earth_radius = 6371.0
mantle_depth = 2890.0

#=====================================================================
#=====================================================================
def usage():
    '''print usage message and exit'''

    print('''plot_Rhea_xsection_compression.py [-e] configuration_file.cfg 

    TODO: add usage documentation
''')

    sys.exit(0)
#====================================================================
#====================================================================
def main() :
    '''main function of the script'''

    print( "\nScript: plot_Rhea_xsection_compression.py" )

    # Get run name
    run_name = sys.argv[1]
    print( "Run name:", run_name )
    print( "" )

    # Note: ordering in xsections_list is important!!
    xsections_list = ["Aleutians1","Aleutians2","Chile1","Japan1","Kermadec1","Kurile1",
                     "Marianas1","Marianas2","NewHebrides1","Peru1","Peru2", 
                     "Sandwich1","Tonga1","Tonga2"]

    xsections_list = ["Aleutians1","Aleutians2","Chile1","Japan1"]

    xsections_list = ["Aleutians1"]

    data_file = "xsections_converted.txt"

    # Set GMT parameters
    cmd = "gmtset HEADER_FONT_SIZE 14"
    os.system(cmd)
    cmd = "gmtset LABEL_FONT_SIZE 12"
    os.system(cmd)
    cmd = "gmtset ANNOT_FONT_SIZE 12"
    os.system(cmd)
    cmd = "gmtset PAGE_ORIENTATION portrait"
    os.system(cmd)
    cmd = "gmtset MEASURE_UNIT inch"
    os.system(cmd)

    equator_normal,domain,subd_dir = Core_Rhea.get_xsection_data(data_file)

    for nr,xsection in enumerate(xsections_list):

        print( "PROCESSING: xsection =", xsection, " ; nr = ", nr )

        # Plot parameters
        psfile = "%s_%s_compression.ps" % (run_name, xsection)
        region = "0/360/%s/%s" % (earth_radius - mantle_depth, earth_radius)

        # Get domain of xsection
        print( "    Get domain ..." )
        lon1 = domain[nr][0]
        lat1 = domain[nr][1]
        lon2 = domain[nr][2]
        lat2 = domain[nr][3]
        print( "    lon1,lat1,lon2,lat2:", lon1, lat1, lon2, lat2 )
        direction = subd_dir[nr]

        # CMT data dir
        CMT_dir = "../CMT/"
        print( 'setting CMT_dir = ', CMT_dir )

        #====================================================================

        # Project Rhea data
        print( "    Projecting Rhea data ..." )
        Core_Rhea.project_Rhea_compression(run_name,xsection,lon1,lat1,lon2,lat2,direction)

        # Process compression from Rhea
        ce_xyz = "%s_annulus_ce.xyz" % (xsection)
        ce_grd = "%s_annulus_ce.grd" % (xsection)
        ce_grd_masked = "%s_annulus_ce_masked.grd" % (xsection)
        cn_xyz = "%s_annulus_cn.xyz" % (xsection)
        cn_grd = "%s_annulus_cn.grd" % (xsection)
        cn_grd_masked = "%s_annulus_cn_masked.grd" % (xsection)
        cr_xyz = "%s_annulus_cr.xyz" % (xsection)
        cr_grd = "%s_annulus_cr.grd" % (xsection)
        cr_grd_masked = "%s_annulus_cr_masked.grd" % (xsection)
        T_xyz = "%s_annulus_T.xyz" % (xsection)
        T_grd = "%s_annulus_T.grd" % (xsection)
        res_lon = 0.05
        res_r = 5.0

        print( "    Processing %s to %s ..." % (ce_xyz,ce_grd) )
        cmd = "blockmean %s -R%s -I%s/%s > mean_ce.xyz" % (ce_xyz,region,res_lon,res_r)
        os.system(cmd)
        cmd = "surface mean_ce.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,ce_grd)
        os.system(cmd)
        
        print( "    Processing %s to %s ..." % (cr_xyz,cr_grd) )
        cmd = "blockmean %s -R%s -I%s/%s > mean_cr.xyz" % (cr_xyz,region,res_lon,res_r)
        os.system(cmd)
        cmd = "surface mean_cr.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,cr_grd)
        os.system(cmd)

        print( "    Processing %s to %s ..." % (T_xyz,T_grd) )
        cmd = "blockmean %s -R%s -I%s/%s > mean_T.xyz" % (T_xyz,region,res_lon,res_r)
        os.system(cmd)
        cmd = "surface mean_T.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,T_grd)
        os.system(cmd)
        print( "" )

        #====================================================================

        # Project CMT data
        print( "    Projecting CMT data ..." )
        scaling = 0.12
        Core_Rhea.project_CMT_compression(CMT_dir,xsection,scaling,lon1,lat1,lon2,lat2,direction)

        # Process compression from CMT
        CMT_ce_xyz = "%s_CMT_ce.xyz" % (xsection)
        CMT_ce_grd = "%s_CMT_ce.grd" % (xsection)
        CMT_ce_grd_masked = "%s_CMT_ce_masked.grd" % (xsection)
        CMT_cn_xyz = "%s_CMT_cn.xyz" % (xsection)
        CMT_cn_grd = "%s_CMT_cn.grd" % (xsection)
        CMT_cn_grd_masked = "%s_CMT_cn_masked.grd" % (xsection)
        CMT_cr_xyz = "%s_CMT_cr.xyz" % (xsection)
        CMT_cr_grd = "%s_CMT_cr.grd" % (xsection)
        CMT_cr_grd_masked = "%s_CMT_cr_masked.grd" % (xsection)

        print( "    Processing %s to %s ..." % (CMT_ce_xyz,CMT_ce_grd) )
        cmd = "blockmean %s -R%s -I%s/%s > mean_ce.xyz" % (CMT_ce_xyz,region,res_lon,res_r)
        os.system(cmd)
        cmd = "surface mean_ce.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,CMT_ce_grd)
        os.system(cmd)

        print( "    Processing %s to %s ..." % (CMT_cr_xyz,CMT_cr_grd) )
        cmd = "blockmean %s -R%s -I%s/%s > mean_cr.xyz" % (CMT_cr_xyz,region,res_lon,res_r)
        os.system(cmd)
        cmd = "surface mean_cr.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,CMT_cr_grd)
        os.system(cmd)
        print( "" )

        #====================================================================

        # Mask temperature to only show plates and slabs
        print( "    Masking temperature outside of slab ..." )
        T_clip = 0.8
        cmd = "grdclip %s -GT_clip.grd -Sb%s/1.0 -Sa%s/0.0" % (T_grd, T_clip, T_clip)
        os.system(cmd)

        # Mask out lower mantle from temperature 
        MF = open("mask_LM.xy","w")
        MF.write( "0.0 %s\n" % (earth_radius - 665.0) )
        MF.write( "360.0 %s\n" % (earth_radius - 665.0) )
        MF.write( "360.0 %s\n" % (mantle_depth) )
        MF.write( "0.0 %s\n" % (mantle_depth) )
        MF.close()
        cmd = "grdmask mask_LM.xy -R%s -Gmask.grd -N1/1/0 -I%s/%s" % (region,res_lon,res_r)
        os.system(cmd)
        cmd = "grdmath mask.grd T_clip.grd MUL = T_masked.grd"
        os.system(cmd)

        # Use temperature mask on Rhea compression
        print( "    Masking Rhea compression outside of slab ..." )
        cmd = "grdmath %s T_masked.grd MUL = %s" % (ce_grd,ce_grd_masked)
        os.system(cmd)
        cmd = "grdmath %s T_masked.grd MUL = %s" % (cr_grd,cr_grd_masked)
        os.system(cmd)

        # Use temperature mask on CMT compression
        print( "    Masking CMT compression outside of slab ..." )
        cmd = "grdmath %s T_masked.grd MUL = %s" % (CMT_ce_grd,CMT_ce_grd_masked)
        os.system(cmd)
        cmd = "grdmath %s T_masked.grd MUL = %s" % (CMT_cr_grd,CMT_cr_grd_masked)
        os.system(cmd)

        # Use temperature mask on compression misfit
        #print( "    Masking compression misfit field outside of slab ..." )
        #cmd = "grdmath %s T_masked.grd MUL = %s" % (misfit_grd,misfit_grd_masked)
        #os.system(cmd)

        xmin,xmax = Core_Rhea.find_new_domain(lon1,lat1,lon2,lat2)
        center = 0.5 * (xmin + xmax)

        # Sample compression for plotting
        plot_pts = "plot_points.xyz"
        plot_res_lon = 0.4   
        plot_res_r = 25.0
        scaling = 0.12
        Core_Rhea.sample_points(plot_pts,plot_res_lon,plot_res_r)
        print( "    Sampling Rhea compression for plotting ..." )
        compression_sampled = Core_Rhea.sample_compression(xsection,ce_grd_masked,cn_grd_masked,cr_grd_masked,plot_pts,scaling,0,center,direction)
        #print( "    Sampling CMT compression for plotting ..." )
        #CMT_compression_sampled = Core_Rhea.sample_compression(xsection,CMT_ce_grd_masked,CMT_cn_grd_masked,CMT_cr_grd_masked,plot_pts,scaling,1,center,direction)
        plot_pts = "%s_CMT_points.xyz" % (xsection)
        compression_sampled_onCMT = Core_Rhea.sample_compression(xsection,ce_grd,cn_grd,cr_grd,plot_pts,scaling,2,center,direction)
        print( "" )

        CMT_compression = "%s_CMT.xyV" % (xsection)

        #====================================================================

        # Compute misfit between Rhea and CMT compression
        #print( "    Computing averaged misfit between Rhea and CMT compression ..." )
        #compression_misfit = Core_Rhea.inner_prod(xsection,ce_grd,cn_grd,cr_grd,CMT_ce_grd,CMT_cn_grd,CMT_cr_grd,res_lon,res_r)
        # Use angle difference between Tonga1_compression_sampled_onCMT.xyV and CMT_Tonga1_compression_sampled.xyV
        #compression_misfit = "%s_compression_misfit.xyz" % (xsection)
        #misfit_grd = "%s_compression_misfit.grd" % (xsection)
        #misfit_grd_masked = "%s_compression_misfit_masked.grd" % (xsection)
        #cmd = "blockmean %s -R%s -I%s/%s > mean_misfit.xyz" % (compression_misfit,region,res_lon,res_r)
        #os.system(cmd)
        #cmd = "surface mean_misfit.xyz -R -I%s/%s -T0.2 -Gmisfit.grd" % (res_lon,res_r)
        #os.system(cmd)
        #cmd = "grdfilter misfit.grd -G%s -D0 -Fc0.5" % (misfit_grd)
        #os.system(cmd)


        #ave_misfit = Core_Rhea.compute_misfit(run_name,xsection,CMT_compression,compression_sampled_onCMT)
        #misfit_file = "%s_%s_compression_ave_misfit.dat" % (run_name, xsection)
        #MF = open(misfit_file,"w")
        #MF.write( "%s     %g" % (xsection,ave_misfit) )
        #MF.close()
        #print( "" )

        #====================================================================

        xmin,xmax = Core_Rhea.find_new_domain(lon1,lat1,lon2,lat2)
        center = 0.5 * (xmin + xmax)

        # Plot background
        print( "    Plotting background ..." )
        cptfile = "mask.cpt"
        cmd = "makecpt -Cgray -T0/4/0.1 -I -D > %s" % (cptfile)

        #cptfile = "misfit.cpt"
        #cmd = "makecpt -Chot -T0/1/0.01 -I -Z -D > %s" % (cptfile)

        os.system(cmd)
        
        region = "%s/%s/%s/%s" % (xmin, xmax, earth_radius - 700.0, earth_radius)
        if (direction == "R"): # Subduction to the right
            proj = "Pa6i/%sz" % (center)
        else:                  # Subduction to the left
            proj = "P6i/%sz" % (center - 90.0)
        labels = "a2f1:'':/a200f50:'':WEsN:.'%s %s compression':" % (run_name,xsection)
        cmd = "psbasemap -R%s -J%s -B%s -Y2.0 -K > %s" % (region, proj, labels, psfile)
        os.system(cmd)
        cmd = "grdimage -R -J T_masked.grd -Q -C%s -O -K >> %s" % (cptfile,psfile)
        #cmd = "grdimage -R -J %s -Q -C%s -O -K >> %s" % (misfit_grd_masked,cptfile,psfile)
        os.system(cmd)
        contour = 1.0
        cmd = "grdcontour -R -J T_masked.grd -C%s -W1p -K -O >> %s" % (contour,psfile)
        #os.system(cmd)

        #cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Gred -O -K >> %s" % (CMT_compression_sampled,psfile)
        #os.system(cmd)

        # Plot Rhea compression
        print( "    Plotting Rhea compression ..." )
        cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Gblue -W1,blue -O -K >> %s" % (compression_sampled,psfile)
        os.system(cmd)

        # Plot CMT compression
        print( "    Plotting CMT compression ..." )
        cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Gred -W1,red -O -K >> %s" % (CMT_compression,psfile)
        os.system(cmd)
        cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Ggreen -W1,green -O -K >> %s" % (compression_sampled_onCMT,psfile)
        os.system(cmd)

        # Print label and compression scale
        region = "0.0/8.5/0.0/11.0"
        proj = "x1.0"
        LT=open("label.txt","w")
        #LT.write( "%s %s 12 0 0 1 Ave misfit: %8.5g degrees\n" % (4.5, 1.0, ave_misfit) )
        LT.write( "%s %s 12 0 0 1 Rhea\n" % (5.0, 0.8) )
        LT.write( "%s %s 12 0 0 1 CMT\n" % (5.0, 0.6) )
        LT.write( "%s %s 12 0 0 1 Rhea interpolated\n" % (5.0, 0.4) )
        LT.close()
        cmd="pstext label.txt -J%s -R%s -G0 -W255 -X-2.5 -Y-1.2 -O -K >> %s" % (proj,region,psfile)
        os.system(cmd)

        XYV=open("scale.xyV","w")
        XYV.write( "%s %s 90. 0.2" % (4.5, 0.85) )
        XYV.close()
        cmd="psxy scale.xyV -J -R -SV0.005/0/0n1.0 -Gblue -W1,blue -O -K >> %s" % (psfile)
        os.system(cmd)

        XYV=open("scale.xyV","w")
        XYV.write( "%s %s 90. 0.2" % (4.5, 0.65) )
        XYV.close()
        cmd="psxy scale.xyV -J -R -SV0.005/0/0n1.0 -Gred -W1,red -O -K >> %s" % (psfile)
        os.system(cmd)

        XYV=open("scale.xyV","w")
        XYV.write( "%s %s 90. 0.2" % (4.5, 0.45) )
        XYV.close()
        cmd="psxy scale.xyV -J -R -SV0.005/0/0n1.0 -Ggreen -W1,green -O >> %s" % (psfile)
        os.system(cmd)

        #====================================================================

        # Convert file to pdf
        print( "\n    Converting file to pdf ..." )
        cmd = "ps2raster %s -A -Tf -E200" % (psfile)
        os.system(cmd)

        cmd = "rm *.grd *.txt *.xyz *.xyV *.xy"
        #os.system(cmd)
        print( "" )

    print( "" )
    print( "Done!" )
    print( "" )

#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# example config.cfg file for plot_Rhea_xsection_compression.py
# ... 
# 
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

