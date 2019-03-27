#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Laura Alisic, Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This module holds functions related to processing data from the  Rhea model'''

#=====================================================================
#=====================================================================
import datetime, os, pprint, random, re, subprocess, string, sys, traceback, math
import numpy as np

import Core_GMT, Core_Util 

# load the system defaults
sys.path.append( os.path.dirname(__file__) + '/geodynamic_framework_data/')
import geodynamic_framework_configuration_types

#=====================================================================
# Global variables
verbose = False 


# FIXME : replace with numpy values
r2d = 180.0/np.pi
d2r = 1.0/r2d

# FIXME : replace with values set in a Core_ Module ? 
earth_radius = 6371.0
mantle_depth = 2890.0

#====================================================================
#====================================================================
def get_xsection_data(data_file):
    '''Get cross section data from a file '''

    # From /Users/alisic/Research/Results/analysis_global_models/maps/xsections_converted.txt
    equator_normal = []
    domain = []
    subd_dir = []

    data_infile = open(data_file,"r")
    line_nr = 1
    while 1:
        line = data_infile.readline()
        if (line):
            if (line_nr > 1):
                code,lon1,lat1,lon2,lat2,normalx,normaly,normalz,direction = line.split()
                equator_normal.append([float(normalx),float(normaly),float(normalz)])
                lon1new = float(lon1)
                if (lon1new < 0): lon1new += 360.0
                lon2new = float(lon2)
                if (lon2new < 0): lon2new += 360.0
                domain.append([float(lon1new),float(lat1),float(lon2new),float(lat2)])
                subd_dir.append(direction)
            line_nr += 1
        else:
            break

    data_infile.close()

    return equator_normal,domain,subd_dir

#====================================================================
#====================================================================
def project_Rhea_compression(run_name,xsection,lon1,lat1,lon2,lat2,direction):
    '''project the Rhea compression '''

    # Open input and output files
    infilename = "xsections_csv/%s_annulus.csv" % (xsection)
    infile = open(infilename,"r")
    spher_file = "%s_comp_spher_coord.xyz" % (xsection)
    spher_out = open(spher_file,"w")

    # Convert compression axes and coords to spherical, write to file
    line_nr = 1
    for line in infile.readlines():
        if (line_nr > 1):
            # Input is in cartesian
            visc,visc_dim,IIinv,IIinv_dim,u1,u2,u3,C1,C2,C3,T1,T2,T3,stress_dim,dissip_dim,x1,x2,x3 = line.split(",")

            # Convert coords from cartesian to spherical: lon, colat, r
            s1,s2,s3 = Core_Util.cart2spher_coord(x1,x2,x3)

            # Convert vectors from cartesian to spherical
            # v1: r component; v2: n component; v3: e component
            C1new,C2new,C3new = Core_Util.cart2spher_vector(C1,C2,C3,x1,x2,x3)
            lat = 90.0 - r2d*s2
            lon = 180.0 + r2d*s3
            C2new = -1.0*C2new
            r = s1
                
            # Write to file 
            spher_out.write("%g %g %g %g %g %g\n" % (lon,lat,r,C1new,C2new,C3new) )

        line_nr += 1

    infile.close()
    spher_out.close()

    # Convert xsectino start and end to cartesian
    phi1 = lon1 * d2r
    theta1 = (90.0 - lat1) * d2r
    gamma1 = lat1 * d2r
    phi2 = lon2 * d2r
    theta2 = (90.0 - lat2) * d2r
    gamma2 = lat2 * d2r
    r = 1.0
    x1,y1,z1 = Core_Util.spher2cart_coord(r,theta1,phi1)
    x2,y2,z2 = Core_Util.spher2cart_coord(r,theta2,phi2)

    # In-plane unit vector
    pvxyz = np.matrix([x2-x1,y2-y1,z2-z1]).T
    puvxyz = pvxyz / np.linalg.norm(pvxyz)

    # Open new input and output files
    spher_out = open(spher_file,"r")
    cr_file = "%s_annulus_cr.xyz" % (xsection)
    cr_out = open(cr_file,"w")
    cn_file = "%s_annulus_cn.xyz" % (xsection)
    cn_out = open(cn_file,"w")
    ce_file = "%s_annulus_ce.xyz" % (xsection)
    ce_out = open(ce_file,"w")

    # Loop over lines in compression file, write out projected components and coords
    for line in spher_out.readlines():

        # Input is spherical
        l1,l2,r,C1,C2,C3 = line.split()

        # Convert to colat, get lon in right domain
        lon = float(l1)
        if lon < 0: lon += 360.0
        phi = lon * d2r
        lat = float(l2)
        theta = (90.0 - lat) * d2r
        gamma = lat * d2r

        # Convert coords from spherical to cartesian
        rst = float(r) * np.sin(theta)
        x = rst * np.cos(phi)
        y = rst * np.sin(phi)
        z = float(r) * np.cos(theta)

        # Projection matrix to convert velo from spherical to cartesian  
        sint = np.sin(theta)
        sinf = np.sin(phi)
        cost = np.cos(theta)
        cosf = np.cos(phi)
        Crtp2xyz = np.matrix([[cost * cosf, -sinf, sint * cosf],
                              [cost * sinf, cosf , sint * sinf],
                              [   -sint   ,  0   ,    cost    ]])

        # Column vectors
        Crtp = np.matrix([float(C2), float(C3), float(C1)]).T
        Cxyz = np.dot(Crtp2xyz,Crtp)

        # Radial unit vector (in-plane for this point; origin is 3rd point)
        rvxyz = -np.matrix([x,y,z]).T
        ruvxyz = rvxyz / np.linalg.norm(rvxyz)

        # Unit normal
        nvxyz = np.cross(ruvxyz.T,puvxyz.T)
        nuvxyz = nvxyz.T / np.linalg.norm(nvxyz)

        # Tangential unit vector (in-plane for this point)
        tvxyz = np.cross(nuvxyz.T,ruvxyz.T)
        tuvxyz = tvxyz.T / np.linalg.norm(tvxyz)

        # Tangential and radial components
        rcomp = np.dot(ruvxyz.T,Cxyz)
        tcomp = np.dot(tuvxyz.T,Cxyz)

        # Depth
        depth = float(r) * earth_radius

        # Dist to start of xsection using Haversine formula
        h = 0.5 * (1.0 - np.cos(gamma - gamma1)) + 0.5 * np.cos(gamma) * np.cos(gamma1) * (1.0 - np.cos(phi - phi1))
        dist = 2.0 * np.arcsin (np.sqrt(h)) * r2d
        if (direction == 'R'):
            if (lon < lon1): dist = -dist #+ 360.0
        else:
            if (lon > lon1): dist = -dist #+ 360.0

        # Write to file
        cr_out.write("%g   %g   %g\n" % (dist,depth,rcomp) )
        cn_out.write("%g   %g   %g\n" % (dist,depth,float(C3)) )
        ce_out.write("%g   %g   %g\n" % (dist,depth,tcomp) )

    spher_out.close()
    cr_out.close()
    cn_out.close()
    ce_out.close()

    # Convert temperature coords to spherical, write to file
    Tinfilename = "xsections_csv/%s_temp_annulus.csv" % (xsection)
    Tinfile = open(Tinfilename,"r")
    T_file = "%s_annulus_T.xyz" % (xsection)
    T_out  = open(T_file,"w")
    line_nr = 1
    while 1:
        Tline = Tinfile.readline()
        if (Tline):
            if (line_nr > 1):
                # Input is in cartesian
                T,y1,y2,y3 = Tline.split(",")

                # Convert coords from cartesian to spherical
                s1,s2,s3 = Core_Util.cart2spher_coord(y1,y2,y3)
                lat = 90.0 - r2d*s2
                lon = 180.0 + r2d*s3
                r = s1
                depth = r * earth_radius

                if lon < 0: lon += 360.0
                phi = lon * d2r
                theta = (90.0 - lat) * d2r
                gamma = lat * d2r 

                # Dist to start of xsection using Haversine formula
                h = 0.5 * (1.0 - np.cos(gamma - gamma1)) + 0.5 * np.cos(gamma) * np.cos(gamma1) * (1.0 - np.cos(phi - phi1))
                dist = 2.0 * np.arcsin (np.sqrt(h)) * r2d
                if (direction == 'R'):
                    if (lon < lon1): dist = -dist #+ 360.0
                else:
                    if (lon > lon1): dist = -dist #+ 360.0

                T_out.write("%g   %g   %g\n" % (dist,depth,float(T)) )

            line_nr += 1
        else:
            break

    Tinfile.close()
    T_out.close()

    return 1

#====================================================================
#====================================================================

def project_CMT_compression(CMT_dir,xsection,scaling,lon1,lat1,lon2,lat2,direction):
    '''Project the Centroid Moment Tensor'''

    # Open input and output files
    infilename = "%s/%s_CMTs.dat" % (CMT_dir,xsection)
    infile = open(infilename,"r")
    spher_file = "%s_spher_CMT_coord.xyz" % (xsection)    
    spher_out = open(spher_file,"w")
    cr_file = "%s_CMT_cr.xyz" % (xsection)
    cr_out = open(cr_file,"w")
    cn_file = "%s_CMT_cn.xyz" % (xsection)
    cn_out = open(cn_file,"w")
    ce_file = "%s_CMT_ce.xyz" % (xsection)
    ce_out = open(ce_file,"w")
    CMT_file = "%s_CMT.xyV" % (xsection)
    CMT_out = open(CMT_file,"w")
    CMT_points_file = "%s_CMT_points.xyz" % (xsection)
    CMT_points_out = open(CMT_points_file,"w")

    # Convert compression axes and coords to spherical, write to file
    line_nr = 1
    for line in infile.readlines():
        if (line_nr > 1):
            # Input is in cartesian
            lon,lat,depth,pl3,az3,xAproj,rA,xBproj,rB = line.split(" ")

            pl = float(pl3) * d2r
            az = float(az3) * d2r
            C1 = np.sin(pl)
            C2 = np.cos(pl) * np.cos(az)
            C3 = np.cos(pl) * np.sin(az)
            norm = np.sqrt(C1**2 + C2**2 + C3**2)
            C1new = C1 / norm
            C2new = C2 / norm
            C3new = C3 / norm
            r = float(depth) / earth_radius
 
            spher_out.write("%g %g %g %g %g %g\n" % (float(lon),float(lat),r,C1new,C2new,C3new) )

        line_nr += 1

    infile.close()
    spher_out.close()

    # Convert xsection start and end to cartesian
    phi1 = lon1 * d2r
    theta1 = (90.0 - lat1) * d2r
    gamma1 = lat1 * d2r
    phi2 = lon2 * d2r
    theta2 = (90.0 - lat2) * d2r
    gamma2 = lat2 * d2r
    r = 1.0
    x1,y1,z1 = Core_Util.spher2cart_coord(r,theta1,phi1)
    x2,y2,z2 = Core_Util.spher2cart_coord(r,theta2,phi2)

    # In-plane unit vector
    pvxyz = np.matrix([x2-x1,y2-y1,z2-z1]).T
    puvxyz = pvxyz / np.linalg.norm(pvxyz)

    # Open new input and output files
    spher_out = open(spher_file,"r")
    cr_file = "%s_CMT_cr.xyz" % (xsection)
    cr_out = open(cr_file,"w")
    cn_file = "%s_CMT_cn.xyz" % (xsection)
    cn_out = open(cn_file,"w")
    ce_file = "%s_CMT_ce.xyz" % (xsection)
    ce_out = open(ce_file,"w")
    CMT_points_file = "%s_CMT_points.xyz" % (xsection)
    CMT_points_out = open(CMT_points_file,"w")
    CMT_file = "%s_CMT.xyV" % (xsection)
    CMT_out = open(CMT_file,"w")

    # Loop over lines in compression file, write out projected components and coords
    for line in spher_out.readlines():

        # Input is spherical
        l1,l2,r,C1,C2,C3 = line.split()

        # Convert to colat, get lon in right domain
        lon = float(l1)
        if lon < 0: lon += 360.0
        phi = lon * d2r
        lat = float(l2)
        theta = (90.0 - lat) * d2r
        gamma = lat * d2r

        # Convert coords from spherical to cartesian
        rst = float(r) * np.sin(theta)
        x = rst * np.cos(phi)
        y = rst * np.sin(phi)
        z = float(r) * np.cos(theta)

        # Projection matrix to convert velo from spherical to cartesian  
        sint = np.sin(theta)
        sinf = np.sin(phi)
        cost = np.cos(theta)
        cosf = np.cos(phi)
        Crtp2xyz = np.matrix([[cost * cosf, -sinf, sint * cosf],
                              [cost * sinf, cosf , sint * sinf],
                              [   -sint   ,  0   ,    cost    ]])

        # Column vectors
        Crtp = np.matrix([float(C2), float(C3), float(C1)]).T
        Cxyz = np.dot(Crtp2xyz,Crtp)

        # Radial unit vector (in-plane for this point; origin is 3rd point)
        rvxyz = -np.matrix([x,y,z]).T
        ruvxyz = rvxyz / np.linalg.norm(rvxyz)

        # Unit normal
        nvxyz = np.cross(ruvxyz.T,puvxyz.T)
        nuvxyz = nvxyz.T / np.linalg.norm(nvxyz)

        # Tangential unit vector (in-plane for this point)
        tvxyz = np.cross(nuvxyz.T,ruvxyz.T)
        tuvxyz = tvxyz.T / np.linalg.norm(tvxyz)

        # Tangential and radial components
        rcomp = np.dot(ruvxyz.T,Cxyz)
        tcomp = np.dot(tuvxyz.T,Cxyz)

        # Depth
        depth = float(r) * earth_radius

        # Dist to start of xsection using Haversine formula
        h = 0.5 * (1.0 - np.cos(gamma - gamma1)) + 0.5 * np.cos(gamma) * np.cos(gamma1) * (1.0 - np.cos(phi - phi1))
        dist = 2.0 * np.arcsin (np.sqrt(h)) * r2d
        if (direction == 'R'):
            if (lon < lon1): dist = -dist 
        else:
            if (lon > lon1): dist = -dist

        ### HACK!!! To align CMT solutions with slab outline
        if (xsection == 'Aleutians1'):
            offset = -2.0
        elif (xsection == 'Aleutians2'):
            offset = 0.0
        elif (xsection == 'Chile1'):
            offset = 0.0
        elif (xsection == 'Japan1'):
            offset = 0.0
        elif (xsection == 'Kermadec1'):
            offset = -3.0
        elif (xsection == 'Kurile1'):
            offset = -0.5
        elif (xsection == 'Marianas1'):
            offset = -0.5
        elif (xsection == 'Marianas2'):
            offset = -0.5
        elif (xsection == 'NewHebrides1'):
            offset = 0.0
        elif (xsection == 'Peru1'):
            offset = 0.0
        elif (xsection == 'Peru2'):
            offset = 0.0
        elif (xsection == 'Sandwich1'):
            offset = 0.0
        elif (xsection == 'Tonga1'):
            offset = 0.25
        elif (xsection == 'Tonga2'):
            offset = 0.0

        dist += offset

        # Write to file
        cr_out.write("%g   %g   %g\n" % (dist,depth,rcomp) )
        cn_out.write("%g   %g   %g\n" % (dist,depth,float(C3)) )
        ce_out.write("%g   %g   %g\n" % (dist,depth,tcomp) )
        CMT_points_out.write("%g   %g\n" % (dist,depth) )

        # Compute angle for CMT output directly
        if (direction == 'R'):
            angle = np.arctan2(tcomp,rcomp) * r2d
        else:
            angle = np.arctan2(tcomp,-rcomp) * r2d

        ### HACK!!!
        if (xsection == 'Aleutians1'):
            angle = np.arctan2(tcomp,-rcomp) * r2d

        xmin,xmax = find_new_domain(lon1,lat1,lon2,lat2)
        center = 0.5 * (xmin + xmax)
        if angle < 0: angle += 360.0

        if (direction == 'R'):
            angle = angle + dist - center
        else:
            angle = angle - dist + center

        length = scaling

        CMT_out.write("%g   %g   %g   %g\n" % (dist,depth,angle,length) )

    spher_out.close()
    cr_out.close()
    cn_out.close()
    ce_out.close()
    CMT_out.close()
    CMT_points_out.close()

    return 1

#====================================================================
#====================================================================
def find_new_domain(lon1,lat1,lon2,lat2):
    ''' FIXME '''

    gamma1= lat1 * d2r
    gamma2 = lat2 * d2r
    phi1 = lon1 * d2r
    phi2 = lon2 * d2r

    h = 0.5 * (1.0 - np.cos(gamma2 - gamma1)) + 0.5 * np.cos(gamma1) * np.cos(gamma2) * (1.0 - np.cos(phi2 - phi1))
    dist = 2.0 * np.arcsin (np.sqrt(h)) * r2d

    center = 0.5 * dist
    xmin = 0.0
    xmax = dist

    return xmin,xmax

#====================================================================
#====================================================================
def sample_points(plot_pts,plot_res_lon,plot_res_r):
    '''FIXME: '''

    point_file = open(plot_pts,"w")

    depth_start = mantle_depth

    lon = 0.0
    while (lon < 360.0):
        depth = depth_start
        point_file.write("%g    %g\n" % (lon,depth))
        while (depth <= earth_radius):
            depth += plot_res_r
            point_file.write("%g    %g\n" % (lon,depth))
        lon += plot_res_lon

    point_file.close()

    return 1

#====================================================================
#====================================================================
def sample_compression(xsection,ce_grd,cn_grd,cr_grd,plot_pts,scaling,flag,center,direction):
    '''FIXME: '''

    if (flag == 0):
        compression_sampled = "%s_compression_sampled.xyV" % (xsection)
    elif (flag == 1):
        compression_sampled = "CMT_%s_compression_sampled.xyV" % (xsection)
    elif (flag == 2):
        compression_sampled = "%s_compression_sampled_onCMT.xyV" % (xsection)
    compression_file = open(compression_sampled,"w")

    ce_sampled = "ce_sampled.xyV"
    cr_sampled = "cr_sampled.xyV"

    cmd = "grdtrack %s -G%s -fg > %s" % (plot_pts,ce_grd,ce_sampled)
    os.system(cmd)
    cmd = "grdtrack %s -G%s -fg > %s" % (plot_pts,cr_grd,cr_sampled)
    os.system(cmd)
    
    ce_file = open(ce_sampled,"r")
    cr_file = open(cr_sampled,"r")
    while 1:
        line_e = ce_file.readline()
        line_r = cr_file.readline()
        if (line_e):
            e1,e2,e3 = line_e.split()
            r1,r2,r3 = line_r.split()
            lon = float(e1)
            depth = float(e2)
            ce = float(e3)
            cr = float(r3)            

            if (cr != 0):

                if (direction == 'R'):
                    angle = np.arctan2(ce,-cr) * r2d
                else:
                    angle = np.arctan2(-ce,-cr) * r2d

                ### HACK!!!
                if (xsection == 'Aleutians1'):
                    angle = np.arctan2(-ce,-cr) * r2d

                if angle < 0: angle += 360.0

                if (direction == 'R'):
                    angle = angle + lon - center
                else:
                    angle = angle - lon + center

                #length = hypot(cr,ce) / vscale
                length = scaling

                compression_file.write("%g   %g   %g   %g\n" % (lon,depth,angle,length) ) 

        else:
            break
 
    compression_file.close()
    ce_file.close()
    cr_file.close()

    return compression_sampled

#====================================================================
#====================================================================
def inner_prod(xsection,ce_grd,cn_grd,cr_grd,CMT_ce_grd,CMT_cn_grd,CMT_cr_grd,res_lon,res_r):
    ''' FIXME: '''

    compression_misfit = "%s_compression_misfit.xyz" % (xsection)
    compression_file = open(compression_misfit,"w")

    ce_xyz = "ce.xyz"
    cr_xyz = "cr.xyz"
    CMT_ce_xyz = "CMT_ce.xyz"
    CMT_cr_xyz = "CMT_cr.xyz"

    cmd = "grd2xyz %s > %s" % (ce_grd, ce_xyz)
    os.system(cmd)
    cmd = "grd2xyz %s > %s" % (ce_grd, cr_xyz)
    os.system(cmd)
    cmd = "grd2xyz %s > %s" % (CMT_ce_grd, CMT_ce_xyz)
    os.system(cmd)
    cmd = "grd2xyz %s > %s" % (CMT_ce_grd, CMT_cr_xyz)
    os.system(cmd)

    ce_file = open(ce_xyz,"r")
    cr_file = open(cr_xyz,"r")
    CMT_ce_file = open(CMT_ce_xyz,"r")
    CMT_cr_file = open(CMT_cr_xyz,"r")

    while 1:
        line_e = ce_file.readline()
        line_r = cr_file.readline()
        line_Ce = CMT_ce_file.readline()
        line_Cr = CMT_cr_file.readline()

        if (line_e):
            # Read all interpolated data from Rhea compression and CMT compression
            e1,e2,e3 = line_e.split()
            r1,r2,r3 = line_r.split()
            Ce1,Ce2,Ce3 = line_Ce.split()
            Cr1,Cr2,Cr3 = line_Cr.split()

            lon = float(e1)
            depth = float(e2)
            ce = float(e3)
            cr = float(r3)
            CMT_ce = float(Ce3)
            CMT_cr = float(Cr3)

            # Normalize Rhea and CMT compression vectors
            norm = np.sqrt(ce**2 + cr**2)
            CMT_norm = np.sqrt(CMT_ce**2 + CMT_cr**2)
            ce /= norm
            cr /= norm
            CMT_ce /= CMT_norm
            CMT_cr /= CMT_norm

            # Compute inner product
            val = ce*CMT_ce + cr*CMT_cr 

            compression_file.write("%g   %g   %g\n" % (lon,depth,val) )

        else:
            break

    compression_file.close()
    ce_file.close()
    cr_file.close()
    CMT_ce_file.close()
    CMT_cr_file.close()

    return compression_misfit 

#====================================================================
#====================================================================
def compute_misfit(run_name,xsection,CMT_compression,compression_sampled_onCMT):
    ''' TODO '''

    CMT_file = open(CMT_compression,"r")
    Rhea_file = open(compression_sampled_onCMT,"r")
    compression_misfit = "%s_%s_compression_misfit.dat" % (run_name,xsection)
    compression_file = open(compression_misfit,"w")

    count = 0
    misfit_sum = 0
    while 1:
        line = CMT_file.readline()
        line2 = Rhea_file.readline()
        if (line):
            lon1,depth1,angle1,length1 = line.split()
            lon2,depth2,angle2,length2 = line2.split()

            diff = abs(float(angle2) - float(angle1)) % 180.0
            if (diff > 90.0):
                misfit = diff - 90.0
            else:
                misfit = diff

            compression_file.write("%g   %g   %g\n" % (float(lon1),float(depth1),misfit) )
            misfit_sum += misfit
            count += 1   
 
        else:
            break

    if (count > 0):
        ave_misfit = misfit_sum / float(count)
    else:
        ave_misfit = 0.0
    print( 'ave_misfit = ', ave_misfit)

    CMT_file.close()
    Rhea_file.close()
    compression_file.close()

    return ave_misfit

#====================================================================
#====================================================================

#====================================================================
#====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this module'''

    text = '''#=====================================================================
# Core_Rhea_example.cfg
# ... 
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
def test( argv ):
    '''Core_Rhea.py module self test'''
    global verbose
    verbose = True 
    print(now(), 'Core_Rhea.py: test(): sys.argv = ', sys.argv )
    # run the tests 

    # read the defaults
    frame_d = Core_Util.parse_geodynamic_framework_defaults()

    # read the first argument as a .cfg file 
    cfg_d = Core_Util.parse_configuration_file( sys.argv[1] )

    # test  ...

#=====================================================================
#=====================================================================
if __name__ == "__main__":

    # import this module itself
    import Core_Rhea

    if len( sys.argv ) > 1:

        # make the example configuration file 
        if sys.argv[1] == '-e':
            make_example_config_file()
            sys.exit(0)

        # process sys.arv as file names for testing 
        test( sys.argv )
    else:
        # print module documentation and exit
        help(Core_Rhea)
#=====================================================================
#=====================================================================
#=====================================================================
