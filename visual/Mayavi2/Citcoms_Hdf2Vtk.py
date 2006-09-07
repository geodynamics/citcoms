#!/usr/bin/env python

#    Script to generate TVTK files from CitcomS hdf files
#    author: Martin Weier
#    Copyright (C) 2006 California Institue of Technology 
#
#    This program is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.

#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.

#    You should have received a copy of the GNU General Public License
#    along with this program; if not, write to the Free Software
#    Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA


#import scipy
import sys
from datetime import datetime
from getopt import getopt, GetoptError
from pprint import *
from math import *
import tables        #For HDF support
import numpy
import pyvtk
import sys
# defaults

path = "./example0.h5"
vtk_path = "./vtk_output"
vtkfile  = "%s.%d.vtk"

initial = 0
timesteps= None
create_topo = False
create_bottom = False
create_surface = False
create_ascii = False
nx = None
ny = None
nz = None
nx_redu=None
ny_redu=None
nz_redu=None
el_nx_redu = None
el_ny_redu = None
el_nz_redu = None
radius_inner = None
radius_outer = None
nproc_surf = None
#Filehandler to the HDF file
f = None

#####################
polygons3d = []  # arrays containing connectivity information
polygons2d = []
counter=0  #Counts iterations of citcom2vtk  

def print_help():
    print "Program to convert CitcomS HDF to Vtk files.\n"
    print "-p, --path [path to hdf] \n\t Specify input file."
    print "-o, --output [output filename] \n\t Specify the path to the folder for output files."
    print ("-i, --initial [initial timestep] \n\t Specify initial timestep to export. If not \n \
    \t specified script starts exporting from timestep 0.")
    print "-t, --timestep [max timestep] \n\t Specify to which timestep you want to export. If not\n \
    \t specified export all all timestep starting from intial timestep."
    print "-x, --nx_reduce [nx] \n\t Set new nx to reduce output grid."
    print "-y, --ny_reduce [ny] \n\t Set new ny to reduce output grid."
    print "-z, --nz_reduce [nz] \n\t Set new nz to reduce output grid."
    print "-b, --bottom \n\t Set to export Bottom information to Vtk file."
    print "-s, --surface \n\t Set to export Surface information to Vtk file."
    print "-c, --createtopo \n\t Set to create topography information in bottom and surface Vtk file."
    print "-a, --ascii \n\t Create Vtk ASCII encoded files instead if binary."
    print "-h, --help, -? \n\t Print this help."
    

#Iterator for CitcomDataRepresentation(yxz) to VTK(xyz)
def vtk_iter(nx,ny,nz):
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    yield k + i * nz + j * nz * nx

#Reduces the CitcomS grid
def reduce_iter(n,nredu):
    i=0
    n_f=float(n)
    nredu_f=float(nredu)
    fl=(n_f-1)/nredu_f
    redu = 0
    for i in xrange(nredu+1):
        yield int(round(redu))
        redu = redu + fl

#Transform Vectors in Spherical to Cartesian Coordinates 2d                         
#def velocity2cart2d(vel_colat, vel_lon,x , y):
#    x1 = vel_colat*cos(x)*cos(y)-vel_lon*sin(y)
#    y1 = vel_colat*cos(x)*sin(y)+vel_lon*cos(y)
#    z1 = -vel_colat*sin(x)  
#    return x1,y1,z1

#Converts Spherical to Carthesian Coordinates 2d
#def RTF2XYZ2d(vel_colat, vel_lon):
#    x1 = sin(vel_colat)*cos(vel_lon)
#    y1 = sin(vel_colat)*sin(vel_lon)
#    z1 = cos(vel_colat)
#    return x1,y1,z1          

#Transform Vectors in Spherical to Cartesian Coordinates                
def velocity2cart(vel_colat,vel_long,r, x, y, z):
    x1 = r*sin(x)*cos(y)+vel_colat*cos(x)*cos(y)-vel_long*sin(y)
    y1 = r*sin(x)*sin(y)+vel_colat*cos(x)*sin(y)+vel_long*cos(y)
    z1 = r*cos(x)-vel_colat*sin(x)
    return x1, y1, z1


#Converts Spherical to Cartesian Coordinates
def RTF2XYZ(thet, phi, r):
    x = r * sin(thet) * cos(phi)
    y = r * sin(thet) * sin(phi)
    z = r * cos(thet)
    return x, y, z



#Reads Citcom Files and creates a VTK File
def citcom2vtk(t):
    print "Timestep:",t
   
    benchmarkstr = ""
    #Assign create_bottom and create_surface to bottom and surface 
    #to make them valid in methods namespace
    bottom = create_bottom
    surface = create_surface
    
    ordered_points = [] #reset Sequences for points   
    ordered_temperature = []
    ordered_velocity = []
    ordered_visc = []
    
    #Surface and Bottom Points
    #Initialize empty sequences
    surf_vec = []
    botm_vec = []        
    surf_topo = []
    surf_hflux = []
    botm_topo = []
    botm_hflux = []
   
    surf_points = []
    botm_points = []
    
    for capnr in xrange(nproc_surf):
        ###Benchmark Point 1 Start##
        #start = datetime.now()
        ############################
        print "Processing cap",capnr+1,"of",nproc_surf
        cap = f.root._f_getChild("cap%02d" % capnr)
    
        #Information from hdf
        #This information needs to be read only once
        
        hdf_coords = cap.coord[:]
        hdf_velocity = cap.velocity[t]
        hdf_temperature = cap.temperature[t]
        hdf_viscosity = cap.viscosity[t]
        
        ###Benchmark Point 1 Stop##
        #delta = datetime.now() - start
        #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
        
        ###Benchmark Point 2 Start##
        #start = datetime.now()
        ############################
        
        #Create Iterator to change data representation
        nx_redu_iter = reduce_iter(nx,nx_redu)
        ny_redu_iter = reduce_iter(ny,ny_redu)
        nz_redu_iter = reduce_iter(nz,nz_redu)
        #vtk_i = vtk_iter(el_nx_redu,el_ny_redu,el_nz_redu)
        
        # read citcom data - zxy (z fastest)
        for j in xrange(el_ny_redu):
            j_redu = ny_redu_iter.next()
            nx_redu_iter = reduce_iter(nx,nx_redu)
            for i in xrange(el_nx_redu):
                i_redu = nx_redu_iter.next()
                nz_redu_iter = reduce_iter(nz,nz_redu)
                for k in xrange(el_nz_redu):
                    k_redu = nz_redu_iter.next()
                
                    colat, lon, r = map(float,hdf_coords[i_redu,j_redu,k_redu])
                    x_coord, y_coord, z_coord = RTF2XYZ(colat,lon,r)
                    ordered_points.append((x_coord,y_coord,z_coord))
      
                    ordered_temperature.append(float(hdf_temperature[i_redu,j_redu,k_redu]))
                    ordered_visc.append(float(hdf_viscosity[i_redu,j_redu,k_redu]))
                
                    
                    vel_colat, vel_lon , vel_r = map(float,hdf_velocity[i_redu,j_redu,k_redu])
                    x_velo, y_velo, z_velo = velocity2cart(vel_colat,vel_lon,vel_r, colat,lon , r)
                    
                    ordered_velocity.append((x_velo,y_velo,z_velo))                        
        
        
        ##Delete Objects for GC
        del hdf_coords
        del hdf_velocity
        del hdf_temperature
        del hdf_viscosity
        
        ###Benchmark Point 2 Stop##
        #delta = datetime.now() - start
        #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
    
        ###Benchmark Point 3 Start##
        #start = datetime.now()
        ############################

        #Bottom Information from hdf
        if bottom == True:
            try:
                hdf_bottom_coord = cap.botm.coord[:]
                hdf_bottom_heatflux = cap.botm.heatflux[t]
                hdf_bottom_topography = cap.botm.topography[t]
                hdf_bottom_velocity = cap.botm.velocity[t]
            except:
                print "\tCould not find bottom information in file.\n \
                       Set create bottom to false"
                bottom = False
        #Surface Information from hdf
        if surface==True:
            try:
                hdf_surface_coord = cap.surf.coord[:]
                hdf_surface_heatflux = cap.surf.heatflux[t]
                hdf_surface_topography = cap.surf.topography[t]
                hdf_surface_velocity = cap.surf.velocity[t]
            except:
                print "\tCould not find surface information in file.\n \
                       Set create surface to false"
                surface = False
        
        ###Benchmark Point 3 Stop##
        #delta = datetime.now() - start
        #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
        
        
        ###Benchmark Point 4 Start##
        #start = datetime.now()
        ############################
         
        #Compute surface/bottom topography mean
        if create_topo:
            surf_mean=0.0
            botm_mean=0.0
    
            if surface:
                for i in xrange(nx):
                    surf_mean += numpy.mean(hdf_surface_topography[i])
                surf_mean = surf_mean/ny

            if bottom:
                for i in xrange(nx):
                    botm_mean += numpy.mean(hdf_bottom_topography[i])
                botm_mean = botm_mean/nx
        
        
        
        ###Benchmark Point 4 Stop##
        #delta = datetime.now() - start
        #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
    
        ###Benchmark Point 5 Start##
        #start = datetime.now()
        ############################
        
        #Read Surface and Bottom Data   
        if bottom==True or surface == True:
            for i in xrange(nx):
                for j in xrange(ny):
                    
                    
                    if bottom==True:
                        #Bottom Coordinates
                        if create_topo==True:
                            colat, lon = hdf_bottom_coord[i,j]
                            x,y,z = RTF2XYZ(colat,lon,radius_inner+float( (hdf_bottom_topography[i,j]-botm_mean)*(10**21)/(6371000**2/10**(-6))/(3300*10)/1000 ))
                            botm_points.append((x,y,z))
                        else:
                            colat, lon = hdf_bottom_coord[i,j]
                            x,y,z = RTF2XYZ(colat, lon,radius_inner) 
                            botm_points.append((x,y,z))
            
                        #Bottom Heatflux
                        botm_hflux.append(float(hdf_bottom_heatflux[i,j]))
            
                        #Bottom Velocity
                        vel_colat, vel_lon = map(float,hdf_bottom_velocity[i,j])
                        x,y,z = velocity2cart(vel_colat,vel_lon, radius_inner, colat, lon, radius_inner)
                        botm_vec.append((x,y,z))
            
                    if surface==True:
                        #Surface Information
                        if create_topo==True:
                            colat,lon = hdf_surface_coord[i,j]
                            #637100 = Earth radius, 33000 = ?
                            x,y,z = RTF2XYZ(colat,lon,radius_outer+float( (hdf_surface_topography[i,j]-surf_mean)*(10**21)/(6371000**2/10**(-6))/(3300*10)/1000 ))
                            surf_points.append((x,y,z))
                        else:
                            colat, lon = hdf_surface_coord[i,j]
                            x,y,z = RTF2XYZ(colat, lon,radius_outer) 
                            surf_points.append((x,y,z))
            
                        #Surface Heatflux
                        surf_hflux.append(float(hdf_surface_heatflux[i,j]))
            
                        #Surface Velocity
                        vel_colat, vel_lon = map(float,hdf_surface_velocity[i,j])
                        x,y,z = velocity2cart(vel_colat,vel_lon, radius_outer, colat, lon, radius_outer)
                        surf_vec.append((x,y,z))
     
         #del variables for GC
        if bottom==True:
            del hdf_bottom_coord
            del hdf_bottom_heatflux
            del hdf_bottom_velocity
        if surface==True:
            del hdf_surface_coord
            del hdf_surface_heatflux
            del hdf_surface_velocity   
     
        ###Benchmark Point 5 Stop##
        #delta = datetime.now() - start
        #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
    
        ###Benchmark Point 6 Start##
        #start = datetime.now()
        ############################
        
        
##################################################################    
        #Create Connectivity info    
        if counter==0:
            #For 3d Data 
            i=1    #Counts X Direction
            j=1    #Counts Y Direction
            k=1    #Counts Z Direction
    
            for n in xrange((el_nx_redu*el_ny_redu*el_nz_redu)-(el_nz_redu*el_nx_redu)):
                if (i%el_nz_redu)==0:            #X-Values!!!
                    j+=1                 #Count Y-Values
        
                if (j%el_nx_redu)==0:
                    k+=1                #Count Z-Values
                  
                if i%el_nz_redu!=0 and j%el_nx_redu!=0:            #Check if Box can be created
                    #Get Vertnumbers
                    n0 = n+(capnr*(el_nx_redu*el_ny_redu*el_nz_redu))
                    n1 = n0+el_nz_redu
                    n2 = n1+el_nz_redu*el_nx_redu
                    n3 = n0+el_nz_redu*el_nx_redu
                    n4 = n0+1
                    n5 = n4+el_nz_redu
                    n6 = n5+el_nz_redu*el_nx_redu
                    n7 = n4+el_nz_redu*el_nx_redu

                    #Created Polygon Box
                    polygons3d.append([n0,n1,n2,n3,n4,n5,n6,n7]) #Hexahedron VTK Representation
             
                i+=1
        
        
            if bottom==True or surface==True:
                #Connectivity for 2d-Data      
                i=1
                for n in xrange((nx)*(ny) - ny):
                    if i%ny!=0 :
                        n0 = n+(capnr*((nx)*(ny)))
                        n1 = n0+1
                        n2 = n0+ny
                        n3 = n2+1          
                        polygons2d.append([n0,n1,n2,n3])
                    i+=1
        
        ###Benchmark Point 6 Stop##
        #delta = datetime.now() - start
        #benchmarkstr += "%.5lf\n" % (delta.seconds + float(delta.microseconds)/1e6)
    #print benchmarkstr

#################################################################
#Write Data to VTK  
    
    #benchmarkstr = "\n\nIO:\n"
    ###Benchmark Point 7 Start##
    #start = datetime.now()
    ############################
        
    print 'Writing data to vtk...'
    #Surface Points
    if surface==True:
        struct_coords = pyvtk.UnstructuredGrid(surf_points, pixel=polygons2d)                          
        #topo_scal = pyvtk.Scalars(surf_topo,'Surface Topography', lookup_table='default')
        hflux_scal = pyvtk.Scalars(surf_hflux,'Surface Heatflux',lookup_table='default')
        vel_vec = pyvtk.Vectors(surf_vec,'Surface Velocity Vectors')
        ##
        tempdata = pyvtk.PointData(hflux_scal,vel_vec)
        data = pyvtk.VtkData(struct_coords, tempdata,'CitcomS Output %s Timestep %s' % ('surface info',t))
        if create_ascii:
            data.tofile(vtk_path + (vtkfile % ('surface',t)),) 
        else:
            data.tofile(vtk_path + (vtkfile % ('surface',t)),'binary') 
        print "Written Surface information to file"
        
    ###Benchmark Point 7 Stop##
    #delta = datetime.now() - start
    #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
    
    ###Benchmark Point 8 Start##
    #start = datetime.now()
    ############################
    
    if bottom==True:
        #Bottom Points
        struct_coords = pyvtk.UnstructuredGrid(botm_points, pixel=polygons2d)                          
        #topo_scal = pyvtk.Scalars(botm_topo,'Bottom Topography','default')
        hflux_scal = pyvtk.Scalars(botm_hflux,'Bottom Heatflux','default')
        vel_vec = pyvtk.Vectors(botm_vec,'Bottom Velocity Vectors')
        ##
        tempdata = pyvtk.PointData(hflux_scal,vel_vec)
        data = pyvtk.VtkData(struct_coords, tempdata, 'CitcomS Output %s Timestep %s' % ('Bottom info',t))
        if create_ascii:
            data.tofile(vtk_path + (vtkfile % ('bottom',t)))   
        else:
            data.tofile(vtk_path + (vtkfile % ('bottom',t)),'binary')
        print "Written Bottom information to file"

          
    ###Benchmark Point 8 Stop##
    #delta = datetime.now() - start
    #benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
    
 
    
    ###Benchmark Point 9 Start##
    #start = datetime.now()
    
    #General Data
    struct_coords = pyvtk.UnstructuredGrid(ordered_points,hexahedron=polygons3d)
    vel_vec = pyvtk.Vectors(ordered_velocity, 'Velocity Vectors')
    temp_scal = pyvtk.Scalars(ordered_temperature,'Temperature Scalars','default')
    visc_scal = pyvtk.Scalars(ordered_visc,'Viscosity Scalars','default')
    ##
    tempdata = pyvtk.PointData(temp_scal,visc_scal,vel_vec)
    data = pyvtk.VtkData(struct_coords, tempdata, 'CitcomS Output %s Timestep:%d NX:%d NY:%d NZ:%d Radius_Inner:%f' % (path,t,el_nx_redu,el_ny_redu,el_nz_redu,radius_inner))
    ############################
    if create_ascii:
        data.tofile(vtk_path + (vtkfile % ('general',t)))
    else:
        data.tofile(vtk_path + (vtkfile % ('general',t)),'binary')  
    print "Written general data to file"

    ###Benchmark Point 9 Stop##
    #delta = datetime.now() - start
    #benchmarkstr += "%.5lf\n" % (delta.seconds + float(delta.microseconds)/1e6)

    
    #print benchmarkstr
    #print "\n"



# parse command line parameters
def initialize():
    global path
    global vtk_path
    global initial
    global timesteps
    global create_topo 
    global create_bottom 
    global create_surface 
    global create_ascii 
    global nx 
    global ny 
    global nz 
    global nx_redu
    global ny_redu
    global nz_redu
    global el_nx_redu
    global el_ny_redu
    global el_nz_redu
    global radius_inner
    global radius_outer
    global nproc_surf
    global f
    
    try:
        opts, args = getopt(sys.argv[1:], "p:o:i:t:x:y:z:bscah?", ['path=','output=','timestep=','x=','y=','z=','bottom','surface','createtopo','ascii', 'help','?'])
    except GetoptError, msg:
        print "Error: %s" % msg
        sys.exit(1)
    
    if len(opts)<=1:
        print_help()
        sys.exit(0)

    for opt,arg in opts:
        if opt in ('-p','--path'):
            path = arg
    
        if opt in ('-o','--output'):
            vtk_path = arg
    
        if opt in ('-i','--initial'):
            try:
                initial = int(arg)
            except ValueError:
                print "Initial is not a number."
                sys.exit(1)
        if opt in ('-t','--timestep'):
            try:
                timesteps = int(arg)
            except ValueError:
                print "Timestep is not a number."
                sys.exit(1)
        if opt in ('-x','--nx_reduce'):
            try:
                nx_redu = int(arg)
            except ValueError:
                print "NX is not a number."
    
        if opt in ('-y','--ny_reduce'):
            try:
                ny_redu = int(arg)
            except ValueError:
                print "NY is not a number."
    
        if opt in ('-z','--nz_reduce'):
            try:
                nz_redu = int(arg)
            except ValueError:
                print "NZ is not a number."
    
        if opt in ('-b','--bottom'):
            create_bottom = True
            
        if opt in ('-s','--surface'):
            create_surface = True    
        
        if opt in ('-c','--createtopo'):
            create_topo = True
        
        if opt in ('-a','--ascii'):
            create_ascii = True
        
        if opt in ('-h','--help'):
            print_help()
            sys.exit(0)
        if opt == '-?':
            print_help()
            sys.exit(0)
        

    f = tables.openFile(path,'r')

    nx = int(f.root.input._v_attrs.nodex)
    ny = int(f.root.input._v_attrs.nodey)
    nz = int(f.root.input._v_attrs.nodez)

    #If not defined as argument read from hdf
    hdf_timesteps = int(f.root.time.nrows)

    if timesteps==None or timesteps>hdf_timesteps:
        timesteps = hdf_timesteps 
    

    if nx_redu==None:
        nx_redu = nx-1 
    if ny_redu==None:
        ny_redu = ny-1
    if nz_redu==None:
        nz_redu = nz-1
    
    if nx_redu>=nx:
        nx_redu=nx-1
    if ny_redu>=ny:
        ny_redu=ny-1
    if nz_redu>=nz:
        nz_redu=nz-1
    
    el_nx_redu = nx_redu+1
    el_ny_redu = ny_redu+1
    el_nz_redu = nz_redu+1

    radius_inner = float(f.root.input._v_attrs.radius_inner) 
    radius_outer = float(f.root.input._v_attrs.radius_outer)
    nproc_surf = int(f.root.input._v_attrs.nproc_surf)


###############################################################################
def citcoms_hdf2vtk():
    global counter
    #Call initialize to get and set input params
    initialize()
    
    d1 = datetime.now()
    print "Converting Hdf to Vtk"
    print "Initial:",initial, "Timesteps:",timesteps 
    print "NX:",el_nx_redu, "NY:",el_ny_redu, "NZ:", el_nz_redu
    print "Create Bottom: ",create_bottom, " Create Surface: ", create_surface
    print "Create Topography: ", create_topo

    for t in xrange(initial,timesteps):
        start = datetime.now()
        citcom2vtk(t)
        counter+=1
        delta = datetime.now() - start
        print "\t%.3lf sec" % (delta.seconds + float(delta.microseconds)/1e6)

    d2 = datetime.now()
    f.close()
    print "Total: %d seconds" % (d2 - d1).seconds
###############################################################################



if __name__ == '__main__':
    citcoms_hdf2vtk()
