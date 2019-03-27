#!/usr/bin/env python2.6

##################################################
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

import sys, string, os
from  math import *
import numpy as np

r2d = 180.0/pi
d2r = 1.0/r2d
earth_radius = 6371.0
mantle_depth = 2890.0

#==================================================================

def cart2spher_vector(u1,u2,u3,x1,x2,x3):
    x = float(x1)
    y = float(x2)
    z = float(x3)

    r = sqrt(x*x + y*y + z*z)
    eps = 2.220446049250313e-16
    xy = max(eps * r, sqrt(x*x + y*y))

    T11 = x / r
    T21 = x * z / (xy * r)
    T31 = -y / xy
    T12 = y / r
    T22 = y * z / (xy * r)
    T32 = x / xy
    T13 = z / r
    T23 = -xy / r
    T33 = 0.

    v1 = T11*float(u1) + T12*float(u2) + T13*float(u3)
    v2 = T21*float(u1) + T22*float(u2) + T23*float(u3)
    v3 = T31*float(u1) + T32*float(u2) + T33*float(u3)

    return v1, v2, v3

#====================================================================

def cart2spher_coord(x1,x2,x3):

    s1 = sqrt(float(x1)*float(x1) + float(x2)*float(x2) + float(x3)*float(x3))
    s2 = atan2(sqrt(float(x1)*float(x1) + float(x2)*float(x2)),float(x3))
    s3 = atan2(float(x2),float(x1))

    return s1, s2, s3
    
#====================================================================

def spher2cart_coord(s1,s2,s3):

    x1 = s1 * sin(s2) * cos(s3)
    x2 = s1 * sin(s2) * sin(s3)
    x3 = s1 * cos(s2)

    return x1, x2, x3

#====================================================================

def get_xsection_data(data_file):

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

    file.close(data_infile)

    return equator_normal,domain,subd_dir

#====================================================================

def project_Rhea_compression(run_name,xsection,lon1,lat1,lon2,lat2,direction):

    # Open input and output files
    infilename = "../xsections_csv/%s_annulus.csv" % (xsection)
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
            s1,s2,s3 = cart2spher_coord(x1,x2,x3)

            # Convert vectors from cartesian to spherical
            # v1: r component; v2: n component; v3: e component
            C1new,C2new,C3new = cart2spher_vector(C1,C2,C3,x1,x2,x3)
            lat = 90.0 - r2d*s2
            lon = 180.0 + r2d*s3
            C2new = -1.0*C2new
            r = s1
                
            # Write to file 
            spher_out.write("%g %g %g %g %g %g\n" % (lon,lat,r,C1new,C2new,C3new) )

        line_nr += 1

    file.close(infile)
    file.close(spher_out)

    # Convert xsectino start and end to cartesian
    phi1 = lon1 * d2r
    theta1 = (90.0 - lat1) * d2r
    gamma1 = lat1 * d2r
    phi2 = lon2 * d2r
    theta2 = (90.0 - lat2) * d2r
    gamma2 = lat2 * d2r
    r = 1.0
    x1,y1,z1 = spher2cart_coord(r,theta1,phi1)
    x2,y2,z2 = spher2cart_coord(r,theta2,phi2)

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
        rst = float(r) * sin(theta)
        x = rst * cos(phi)
        y = rst * sin(phi)
        z = float(r) * cos(theta)

        # Projection matrix to convert velo from spherical to cartesian  
        sint = sin(theta)
        sinf = sin(phi)
        cost = cos(theta)
        cosf = cos(phi)
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
        h = 0.5 * (1.0 - cos(gamma - gamma1)) + 0.5 * cos(gamma) * cos(gamma1) * (1.0 - cos(phi - phi1))
        dist = 2.0 * asin (sqrt(h)) * r2d
        if (direction == 'R'):
            if (lon < lon1): dist = -dist #+ 360.0
        else:
            if (lon > lon1): dist = -dist #+ 360.0

        # Write to file
        cr_out.write("%g   %g   %g\n" % (dist,depth,rcomp) )
        cn_out.write("%g   %g   %g\n" % (dist,depth,float(C3)) )
        ce_out.write("%g   %g   %g\n" % (dist,depth,tcomp) )

    file.close(spher_out)
    file.close(cr_out)
    file.close(cn_out)
    file.close(ce_out)

    # Convert temperature coords to spherical, write to file
    Tinfilename = "../../../run109/xsections/xsections_csv/%s_temp_annulus.csv" % (xsection)
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
                s1,s2,s3 = cart2spher_coord(y1,y2,y3)
                lat = 90.0 - r2d*s2
                lon = 180.0 + r2d*s3
                r = s1
                depth = r * earth_radius

                if lon < 0: lon += 360.0
                phi = lon * d2r
                theta = (90.0 - lat) * d2r
                gamma = lat * d2r 

                # Dist to start of xsection using Haversine formula
                h = 0.5 * (1.0 - cos(gamma - gamma1)) + 0.5 * cos(gamma) * cos(gamma1) * (1.0 - cos(phi - phi1))
                dist = 2.0 * asin (sqrt(h)) * r2d
                if (direction == 'R'):
                    if (lon < lon1): dist = -dist #+ 360.0
                else:
                    if (lon > lon1): dist = -dist #+ 360.0

                T_out.write("%g   %g   %g\n" % (dist,depth,float(T)) )

            line_nr += 1
        else:
            break

    file.close(Tinfile)
    file.close(T_out)

    return 1

#====================================================================

def project_CMT_compression(CMT_dir,xsection,scaling,lon1,lat1,lon2,lat2,direction):

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
            C1 = sin(pl)
            C2 = cos(pl) * cos(az)
            C3 = cos(pl) * sin(az)
            norm = sqrt(C1**2 + C2**2 + C3**2)
            C1new = C1 / norm
            C2new = C2 / norm
            C3new = C3 / norm
            r = float(depth) / earth_radius
 
            spher_out.write("%g %g %g %g %g %g\n" % (float(lon),float(lat),r,C1new,C2new,C3new) )

        line_nr += 1

    file.close(infile)
    file.close(spher_out)

    # Convert xsection start and end to cartesian
    phi1 = lon1 * d2r
    theta1 = (90.0 - lat1) * d2r
    gamma1 = lat1 * d2r
    phi2 = lon2 * d2r
    theta2 = (90.0 - lat2) * d2r
    gamma2 = lat2 * d2r
    r = 1.0
    x1,y1,z1 = spher2cart_coord(r,theta1,phi1)
    x2,y2,z2 = spher2cart_coord(r,theta2,phi2)

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
        rst = float(r) * sin(theta)
        x = rst * cos(phi)
        y = rst * sin(phi)
        z = float(r) * cos(theta)

        # Projection matrix to convert velo from spherical to cartesian  
        sint = sin(theta)
        sinf = sin(phi)
        cost = cos(theta)
        cosf = cos(phi)
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
        h = 0.5 * (1.0 - cos(gamma - gamma1)) + 0.5 * cos(gamma) * cos(gamma1) * (1.0 - cos(phi - phi1))
        dist = 2.0 * asin (sqrt(h)) * r2d
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
            angle = atan2(tcomp,rcomp) * r2d
        else:
            angle = atan2(tcomp,-rcomp) * r2d

        ### HACK!!!
        if (xsection == 'Aleutians1'):
            angle = atan2(tcomp,-rcomp) * r2d

        xmin,xmax = find_new_domain(lon1,lat1,lon2,lat2)
        center = 0.5 * (xmin + xmax)
        if angle < 0: angle += 360.0

        if (direction == 'R'):
            angle = angle + dist - center
        else:
            angle = angle - dist + center

        length = scaling

        CMT_out.write("%g   %g   %g   %g\n" % (dist,depth,angle,length) )

    file.close(spher_out)
    file.close(cr_out)
    file.close(cn_out)
    file.close(ce_out)
    file.close(CMT_out)
    file.close(CMT_points_out)

    return 1

#====================================================================

def find_new_domain(lon1,lat1,lon2,lat2):

    gamma1= lat1 * d2r
    gamma2 = lat2 * d2r
    phi1 = lon1 * d2r
    phi2 = lon2 * d2r

    h = 0.5 * (1.0 - cos(gamma2 - gamma1)) + 0.5 * cos(gamma1) * cos(gamma2) * (1.0 - cos(phi2 - phi1))
    dist = 2.0 * asin (sqrt(h)) * r2d

    center = 0.5 * dist
    xmin = 0.0
    xmax = dist

    return xmin,xmax

#====================================================================

def sample_points(plot_pts,plot_res_lon,plot_res_r):

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

def sample_compression(xsection,ce_grd,cn_grd,cr_grd,plot_pts,scaling,flag,center,direction):

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
                    angle = atan2(ce,-cr) * r2d
                else:
                    angle = atan2(-ce,-cr) * r2d

                ### HACK!!!
                if (xsection == 'Aleutians1'):
                    angle = atan2(-ce,-cr) * r2d

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

def inner_prod(xsection,ce_grd,cn_grd,cr_grd,CMT_ce_grd,CMT_cn_grd,CMT_cr_grd,res_lon,res_r):

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
            norm = sqrt(ce**2 + cr**2)
            CMT_norm = sqrt(CMT_ce**2 + CMT_cr**2)
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

def compute_misfit(run_name,xsection,CMT_compression,compression_sampled_onCMT):

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
    print ave_misfit

    CMT_file.close()
    Rhea_file.close()
    compression_file.close()

    return ave_misfit

#====================================================================

#      M  A  I  N

#====================================================================

print "\nScript: plot_Rhea_xsection_compression.py"

# Get run name
run_name = sys.argv[1]
print "Run name:", run_name
print ""

# Note: ordering in xsections_list is important!!
xsections_list = "Aleutians1","Aleutians2","Chile1","Japan1","Kermadec1","Kurile1", \
                 "Marianas1","Marianas2","NewHebrides1","Peru1","Peru2", \
                 "Sandwich1","Tonga1","Tonga2"
#xsections_list = "Tonga1",

#data_file = "/Users/alisic/Research/Results/analysis_global_models/maps/xsections_converted.txt"
data_file = "/Users/alisic/Research/Results/analysis_global_models/maps/xsections_converted_temp.txt"

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

equator_normal,domain,subd_dir = get_xsection_data(data_file)

for nr,xsection in enumerate(xsections_list):

    print "%s ..." % (xsection)

    # Plot parameters
    psfile = "%s_%s_compression.ps" % (run_name, xsection)
    region = "0/360/%s/%s" % (earth_radius - mantle_depth, earth_radius)

    # Get domain of xsection
    print "    Get domain ..."
    lon1 = domain[nr][0]
    lat1 = domain[nr][1]
    lon2 = domain[nr][2]
    lat2 = domain[nr][3]
    print "    lon1,lat1,lon2,lat2:", lon1, lat1, lon2, lat2
    direction = subd_dir[nr]

    # CMT data dir
    CMT_dir = "/Users/alisic/Research/CMT/GMT_stressaxes/xsections"
    print ""

    #====================================================================

    # Project Rhea data
    print "    Projecting Rhea data ..."
    project_Rhea_compression(run_name,xsection,lon1,lat1,lon2,lat2,direction)

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

    print "    Processing %s to %s ..." % (ce_xyz,ce_grd)
    cmd = "blockmean %s -R%s -I%s/%s > mean_ce.xyz" % (ce_xyz,region,res_lon,res_r)
    os.system(cmd)
    cmd = "surface mean_ce.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,ce_grd)
    os.system(cmd)
    
    print "    Processing %s to %s ..." % (cr_xyz,cr_grd)
    cmd = "blockmean %s -R%s -I%s/%s > mean_cr.xyz" % (cr_xyz,region,res_lon,res_r)
    os.system(cmd)
    cmd = "surface mean_cr.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,cr_grd)
    os.system(cmd)

    print "    Processing %s to %s ..." % (T_xyz,T_grd)
    cmd = "blockmean %s -R%s -I%s/%s > mean_T.xyz" % (T_xyz,region,res_lon,res_r)
    os.system(cmd)
    cmd = "surface mean_T.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,T_grd)
    os.system(cmd)
    print ""

    #====================================================================

    # Project CMT data
    print "    Projecting CMT data ..."
    scaling = 0.12
    project_CMT_compression(CMT_dir,xsection,scaling,lon1,lat1,lon2,lat2,direction)

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

    print "    Processing %s to %s ..." % (CMT_ce_xyz,CMT_ce_grd)
    cmd = "blockmean %s -R%s -I%s/%s > mean_ce.xyz" % (CMT_ce_xyz,region,res_lon,res_r)
    os.system(cmd)
    cmd = "surface mean_ce.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,CMT_ce_grd)
    os.system(cmd)

    print "    Processing %s to %s ..." % (CMT_cr_xyz,CMT_cr_grd)
    cmd = "blockmean %s -R%s -I%s/%s > mean_cr.xyz" % (CMT_cr_xyz,region,res_lon,res_r)
    os.system(cmd)
    cmd = "surface mean_cr.xyz -R -I%s/%s -T0.5 -G%s" % (res_lon,res_r,CMT_cr_grd)
    os.system(cmd)
    print ""

    #====================================================================

    # Mask temperature to only show plates and slabs
    print "    Masking temperature outside of slab ..."
    T_clip = 0.8
    cmd = "grdclip %s -GT_clip.grd -Sb%s/1.0 -Sa%s/0.0" % (T_grd, T_clip, T_clip)
    os.system(cmd)

    # Mask out lower mantle from temperature 
    MF = open("mask_LM.xy","w")
    print >> MF, "0.0 %s" % (earth_radius - 665.0)
    print >> MF, "360.0 %s" % (earth_radius - 665.0)
    print >> MF, "360.0 %s" % (mantle_depth)
    print >> MF, "0.0 %s" % (mantle_depth)
    MF.close()
    cmd = "grdmask mask_LM.xy -R%s -Gmask.grd -N1/1/0 -I%s/%s" % (region,res_lon,res_r)
    os.system(cmd)
    cmd = "grdmath mask.grd T_clip.grd MUL = T_masked.grd"
    os.system(cmd)

    # Use temperature mask on Rhea compression
    print "    Masking Rhea compression outside of slab ..."
    cmd = "grdmath %s T_masked.grd MUL = %s" % (ce_grd,ce_grd_masked)
    os.system(cmd)
    cmd = "grdmath %s T_masked.grd MUL = %s" % (cr_grd,cr_grd_masked)
    os.system(cmd)

    # Use temperature mask on CMT compression
    print "    Masking CMT compression outside of slab ..."
    cmd = "grdmath %s T_masked.grd MUL = %s" % (CMT_ce_grd,CMT_ce_grd_masked)
    os.system(cmd)
    cmd = "grdmath %s T_masked.grd MUL = %s" % (CMT_cr_grd,CMT_cr_grd_masked)
    os.system(cmd)

    # Use temperature mask on compression misfit
    #print "    Masking compression misfit field outside of slab ..."
    #cmd = "grdmath %s T_masked.grd MUL = %s" % (misfit_grd,misfit_grd_masked)
    #os.system(cmd)

    xmin,xmax = find_new_domain(lon1,lat1,lon2,lat2)
    center = 0.5 * (xmin + xmax)

    # Sample compression for plotting
    plot_pts = "plot_points.xyz"
    plot_res_lon = 0.4   
    plot_res_r = 25.0
    scaling = 0.12
    sample_points(plot_pts,plot_res_lon,plot_res_r)
    print "    Sampling Rhea compression for plotting ..."
    compression_sampled = sample_compression(xsection,ce_grd_masked,cn_grd_masked,cr_grd_masked,plot_pts,scaling,0,center,direction)
    #print "    Sampling CMT compression for plotting ..."
    #CMT_compression_sampled = sample_compression(xsection,CMT_ce_grd_masked,CMT_cn_grd_masked,CMT_cr_grd_masked,plot_pts,scaling,1,center,direction)
    plot_pts = "%s_CMT_points.xyz" % (xsection)
    compression_sampled_onCMT = sample_compression(xsection,ce_grd,cn_grd,cr_grd,plot_pts,scaling,2,center,direction)
    print ""

    #====================================================================

    # Compute misfit between Rhea and CMT compression
    print "    Computing averaged misfit between Rhea and CMT compression ..."
    #compression_misfit = inner_prod(xsection,ce_grd,cn_grd,cr_grd,CMT_ce_grd,CMT_cn_grd,CMT_cr_grd,res_lon,res_r)
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

    CMT_compression = "%s_CMT.xyV" % (xsection)
    ave_misfit = compute_misfit(run_name,xsection,CMT_compression,compression_sampled_onCMT)
    misfit_file = "%s_%s_compression_ave_misfit.dat" % (run_name, xsection)
    MF = open(misfit_file,"w")
    print >> MF, "%s     %g" % (xsection,ave_misfit)
    MF.close()
 
    print ""

    #====================================================================

    xmin,xmax = find_new_domain(lon1,lat1,lon2,lat2)
    center = 0.5 * (xmin + xmax)

    # Plot background
    print "    Plotting background ..."
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
    print "    Plotting Rhea compression ..."
    cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Gblue -W1,blue -O -K >> %s" % (compression_sampled,psfile)
    os.system(cmd)

    # Plot CMT compression
    print "    Plotting CMT compression ..."
    cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Gred -W1,red -O -K >> %s" % (CMT_compression,psfile)
    os.system(cmd)
    cmd = "psxy -R -J %s -SVb0.005/0/0n1.0 -Ggreen -W1,green -O -K >> %s" % (compression_sampled_onCMT,psfile)
    os.system(cmd)

    # Print label and compression scale
    region = "0.0/8.5/0.0/11.0"
    proj = "x1.0"
    LT=open("label.txt","w")
    print >> LT, "%s %s 12 0 0 1 Ave misfit: %8.5g degrees" % (4.5, 1.0, ave_misfit)
    print >> LT, "%s %s 12 0 0 1 Rhea" % (5.0, 0.8)
    print >> LT, "%s %s 12 0 0 1 CMT" % (5.0, 0.6)
    print >> LT, "%s %s 12 0 0 1 Rhea interpolated" % (5.0, 0.4)
    LT.close()
    cmd="pstext label.txt -J%s -R%s -G0 -W255 -X-2.5 -Y-1.2 -O -K >> %s" % (proj,region,psfile)
    os.system(cmd)

    XYV=open("scale.xyV","w")
    print >> XYV, "%s %s 90. 0.2" % (4.5, 0.85)
    XYV.close()
    cmd="psxy scale.xyV -J -R -SV0.005/0/0n1.0 -Gblue -W1,blue -O -K >> %s" % (psfile)
    os.system(cmd)

    XYV=open("scale.xyV","w")
    print >> XYV, "%s %s 90. 0.2" % (4.5, 0.65)
    XYV.close()
    cmd="psxy scale.xyV -J -R -SV0.005/0/0n1.0 -Gred -W1,red -O -K >> %s" % (psfile)
    os.system(cmd)

    XYV=open("scale.xyV","w")
    print >> XYV, "%s %s 90. 0.2" % (4.5, 0.45)
    XYV.close()
    cmd="psxy scale.xyV -J -R -SV0.005/0/0n1.0 -Ggreen -W1,green -O >> %s" % (psfile)
    os.system(cmd)

    #====================================================================

    # Convert file to pdf
    print "\n    Converting file to pdf ..."
    cmd = "ps2raster %s -A -Tf -E200" % (psfile)
    os.system(cmd)

    cmd = "rm *.grd *.txt *.xyz *.xyV *.xy"
    os.system(cmd)
    print ""

print ""
print "Done!"
print ""

# EOF

