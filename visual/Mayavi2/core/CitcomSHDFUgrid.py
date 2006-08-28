from enthought.tvtk.api import tvtk
import tables        #For HDF support
import numpy
from math import *
from datetime import datetime

class CitcomSHDFUgrid:
    
    data = None
    _nx = None
    _ny = None
    _nz = None
    _nx_redu = None
    _ny_redu = None
    _nz_redu = None
    _radius_inner = None
    _radius_outer = None
    timesteps = None
    frequency = None
    
    #Iterator for CitcomDataRepresentation(yxz) to VTK(xyz)
    def vtk_iter(self,nx,ny,nz):
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    yield k + nz * i + nz * nx * j

    #Reduces the CitcomS grid
    def reduce_iter(self,n,nredu):
        i=0
        n_f=float(n)
        nredu_f=float(nredu)
        fl=(n_f-1)/nredu_f
        redu = 0
        for i in xrange(nredu+1):
            yield int(round(redu))
            redu = redu + fl
            
    
    def velocity2cart(self,vel_colat,vel_long,r, x, y, z):
        x1 = r*sin(x)*cos(y)+vel_colat*cos(x)*cos(y)-vel_long*sin(y)
        y1 = r*sin(x)*sin(y)+vel_colat*cos(x)*sin(y)+vel_long*cos(y)
        z1 = r*cos(x)-vel_colat*sin(x)
        return x1, y1, z1


    #Converts Spherical to Cartesian Coordinates
    def RTF2XYZ(self,thet, phi, r):
        x = r * sin(thet) * cos(phi)
        y = r * sin(thet) * sin(phi)
        z = r * cos(thet)
        return x, y, z
    
    
    
    def citcom2vtk(self,t,f,nproc_surf,nx_redu,ny_redu,nz_redu,bottom,surface):
        #Assign create_bottom and create_surface to bottom and surface 
        #to make them valid in methods namespace
        
        benchmarkstr = ""
        fd = open('/home/maweier/benchmark.txt','w')
        
        hexagrid = tvtk.UnstructuredGrid() 
        surfPixelGrid = tvtk.UnstructuredGrid()
        botmPixelGrid = tvtk.UnstructuredGrid()
        
        vtkordered_temp = tvtk.FloatArray()
        vtkordered_velo = tvtk.FloatArray()
        vtkordered_visc = tvtk.FloatArray()
        
        hexagrid.allocate(1,1)
        surfPixelGrid.allocate(1, 1)
        botmPixelGrid.allocate(1,1)
        
        nx = self._nx
        ny = self._ny
        nz = self._nz
        counter = 0
        el_nx_redu = nx_redu + 1
        el_ny_redu = ny_redu + 1
        el_nz_redu = nz_redu + 1
            
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
            d1 = datetime.now()
            ############################
            
            cap = f.root._f_getChild("cap%02d" % capnr)
    
            temp_coords =  [] # reset Coordinates, Velocity, Temperature Sequence
            temp_vel = []     
            temp_temp = []
            temp_visc = []
    
            #Information from hdf
            #This information needs to be read only once
            hdf_coords = cap.coord[:]
        
            hdf_velocity = cap.velocity[t]
            hdf_temperature = cap.temperature[t]
            hdf_viscosity = cap.viscosity[t]
    
           
        
            #Create Iterator to change data representation
            nx_redu_iter = self.reduce_iter(nx,nx_redu)
            ny_redu_iter = self.reduce_iter(ny,ny_redu)
            nz_redu_iter = self.reduce_iter(nz,nz_redu)
      
            vtk_i = self.vtk_iter(el_nx_redu,el_ny_redu,el_nz_redu)
             
            ##Benchmark Point 1 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
            
            ###Benchmark Point 2 Start##
            d1 = datetime.now()
            ############################
            
            # read citcom data - zxy (z fastest)
            for j in xrange(el_ny_redu):
                j_redu = ny_redu_iter.next()
                nx_redu_iter = self.reduce_iter(nx,nx_redu)
                for i in xrange(el_nx_redu):
                    i_redu = nx_redu_iter.next()
                    nz_redu_iter = self.reduce_iter(nz,nz_redu)
                    for k in xrange(el_nz_redu):
                        k_redu = nz_redu_iter.next()
                        thet , phi, r = map(float,hdf_coords[i_redu][j_redu][k_redu])
                        temp_coords.append((thet,phi,r))
                    
                        vel_colat, vel_lon , vel_r = map(float,hdf_velocity[i][j][k])
                        temperature = float(hdf_temperature[i][j][k])
                        visc = float(hdf_viscosity[i][j][k])
                
                        temp_vel.append((vel_colat,vel_lon,vel_r))
                        temp_temp.append(temperature)
                        temp_visc.append(visc)
    
            ##Delete Objects for GC
            del hdf_coords
            del hdf_velocity
            del hdf_temperature
            del hdf_viscosity
            
            ##Benchmark Point 2 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)            
     
            ###Benchmark Point 3 Start##
            d1 = datetime.now()
            ############################
            
            # rearange vtk data - xyz (x fastest).
            for n0 in xrange(el_nz_redu*el_ny_redu*el_nx_redu):
                iter = vtk_i.next()
                #print iter
                #Get Cartesian Coords from Coords
                #zxy Citcom to xyz Vtk
                colat, lon, r = temp_coords[iter]
                x_coord, y_coord, z_coord = self.RTF2XYZ(colat,lon,r)
                ordered_points.append((x_coord,y_coord,z_coord))
      
                #Get Vectors in Cartesian Coords from Velocity
                vel_colat,vel_lon,vel_r = temp_vel[iter]
                x_velo, y_velo, z_velo = self.velocity2cart(vel_colat,vel_lon,vel_r, colat,lon , r)
                ordered_velocity.append((x_velo,y_velo,z_velo))                        
        
                ################################################
                vtkordered_temp.insert_next_tuple1(temp_temp[iter])
                vtkordered_visc.insert_next_tuple1(temp_visc[iter])                                
          
            vtkordered_velo.from_array(ordered_velocity)
          
            ##Delete Unused Object for GC
            del temp_coords
            del temp_vel
            del temp_temp
            del temp_visc

            ##Benchmark Point 3 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)

            ###Benchmark Point 4 Start##
            d1 = datetime.now()
            ############################

           #Bottom Information from hdf
            if bottom == True:
                try:
                    hdf_bottom_coord = cap.botm.coord[:]
                    hdf_bottom_heatflux = cap.botm.heatflux[t]
                    hdf_bottom_topography = cap.botm.topography[t]
                    hdf_bottom_velocity = cap.botm.velocity[t]
                except:
                    bottom = False
            #Surface Information from hdf
            if surface==True:
                try:
                    hdf_surface_coord = cap.surf.coord[:]
                    hdf_surface_heatflux = cap.surf.heatflux[t]
                    hdf_surface_topography = cap.surf.topography[t]
                    hdf_surface_velocity = cap.surf.velocity[t]
                except:
                    surface = False
                    
            ##Benchmark Point 4 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)

            ###Benchmark Point 5 Start##
            d1 = datetime.now()
            ############################
            
            #Compute surface/bottom topography mean
            if bottom==True or surface==True:
                surf_mean=0.0
                botm_mean=0.0
    
                for i in xrange(ny):
                    if surface == True:
                        surf_mean += numpy.mean(hdf_surface_topography[i])
                    if bottom == True:
                        botm_mean += numpy.mean(hdf_bottom_topography[i])
                
                surf_mean = surf_mean/ny
                botm_mean = botm_mean/ny
                #print "Mean Surface:",surf_mean
    
            ##Benchmark Point 5 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
    
            ###Benchmark Point 6 Start##
            d1 = datetime.now()
            ############################
            
            #Read Surface and Bottom Data   
            if bottom==True or surface == True:
                for i in xrange(ny):
                    for j in xrange(nx):
                    
                    
                        if bottom==True:
                            #Bottom Coordinates
                            if create_topo==True:
                                colat, lon = hdf_bottom_coord[i][j]
                                x,y,z = self.RTF2XYZ(colat,lon,radius_inner+float( (hdf_bottom_topography[i][j]-botm_mean)*(10**21)/(6371000**2/10**(-6))/(3300*10)/1000 ))
                                botm_points.append((x,y,z))
                            else:
                                colat, lon = hdf_bottom_coord[i][j]
                                x,y,z = self.RTF2XYZ(colat, lon,radius_inner) 
                                botm_points.append((x,y,z))
            
                            #Bottom Heatflux
                            botm_hflux.append(float(hdf_bottom_heatflux[i][j]))
            
                            #Bottom Velocity
                            vel_colat, vel_lon = map(float,hdf_bottom_velocity[i][j])
                            x,y,z = self.velocity2cart(vel_colat,vel_lon, radius_inner, colat, lon, radius_inner)
                            botm_vec.append((x,y,z))
            
                        if surface==True:
                            #Surface Information
                            if create_topo==True:
                                colat,lon = hdf_surface_coord[i][j]
                                #637100 = Earth radius, 33000 = ?
                                x,y,z = self.RTF2XYZ(colat,lon,radius_outer+float( (hdf_surface_topography[i][j]-surf_mean)*(10**21)/(6371000**2/10**(-6))/(3300*10)/1000 ))
                                surf_points.append((x,y,z))
                            else:
                                colat, lon = hdf_surface_coord[i][j]
                                x,y,z = self.RTF2XYZ(colat, lon,radius_outer) 
                                surf_points.append((x,y,z))
            
                            #Surface Heatflux
                            surf_hflux.append(float(hdf_surface_heatflux[i][j]))
            
                            #Surface Velocity
                            vel_colat, vel_lon = map(float,hdf_surface_velocity[i][j])
                            x,y,z = self.velocity2cart(vel_colat,vel_lon, radius_outer, colat, lon, radius_outer)
                            surf_vec.append((x,y,z))
                
                vtk_botm_vec.from_array(botm_vec)
                vtk_surf_vec.from_array(surf_vec)
                
             #del variables for GC
            if bottom==True:
                del hdf_bottom_coord
                del hdf_bottom_heatflux
                del hdf_bottom_velocity
            if surface==True:
                del hdf_surface_coord
                del hdf_surface_heatflux
                del hdf_surface_velocity    
                
             ##Benchmark Point 6 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf," % (delta.seconds + float(delta.microseconds)/1e6)
            
            ###Benchmark Point 7 Start##
            d1 = datetime.now()
            ############################
##################################################################    
            #Create Connectivity info    
            if counter==0:
                #For 3d Data 
                i=1    #Counts X Direction
                j=1    #Counts Y Direction
                k=1    #Counts Z Direction
    
                for n in xrange(((el_nx_redu*el_ny_redu*el_nz_redu)-(el_nz_redu*el_ny_redu))):
                    if (i%el_nz_redu)==0:            #X-Values!!!
                        j+=1                 #Count Y-Values
        
                    if (j%el_ny_redu)==0:
                        k+=1                #Count Z-Values
                  
                    if i%el_nz_redu!=0 and j%el_ny_redu!=0:            #Check if Box can be created
                        #Get Vertnumbers
                        n0 = n+(capnr*(el_nx_redu*el_ny_redu*el_nz_redu))
                        n1 = n0+1
                        n2 = n1+el_nz_redu
                        n3 = n0+el_nz_redu
                        n4 = n0+(el_ny_redu*el_nz_redu)
                        n5 = n4+1
                        n6 = n4+el_nz_redu+1
                        n7 = n4+el_nz_redu

                        #Created Polygon Box
                        hexagrid.insert_next_cell(12,[n0,n1,n2,n3,n4,n5,n6,n7])
             
                    i+=1
        
                if bottom==True or surface==True:
                    #Connectivity for 2d-Data      
                    i=1
                    for n in xrange((nx)*(ny) - nx):
                        if i%nx!=0 :
                            n0 = n+(capnr*((nx)*(ny)))
                            n1 = n0+1
                            n2 = n0+ny
                            n3 = n2+1          
                            surfPixelGrid.insert_next_cell(8 , [n0,n1,n2,n3])
                            botmPixelGrid.insert_next_cell(8 , [n0,n1,n2,n3])
                        i+=1
    
         ##Benchmark Point 7 Stop##
            delta = datetime.now() - d1
            benchmarkstr += "%.5lf \n" % (delta.seconds + float(delta.microseconds)/1e6)
        
        fd.write(benchmarkstr)
        
        benchmarkstr = '\n\nIO: '
        ###Benchmark Point 1O Start##
        d1 = datetime.now()
        ############################    
        vtkordered_temp.name = 'Temperature'
        hexagrid.point_data.scalars = vtkordered_temp
        vtkordered_velo.name = 'Velocity'
        hexagrid.point_data.vectors = vtkordered_velo
        hexagrid.points = ordered_points
        ##Benchmark Point 1O Stop##
        delta = datetime.now() - d1
        benchmarkstr += "%.5lf" % (delta.seconds + float(delta.microseconds)/1e6)
        fd.write(benchmarkstr)
            
        return hexagrid
            
            
    def initialize(self,filename,timestep,nx_redu,ny_redu,nz_redu,surface,bottom):
      
        hdf=tables.openFile(filename,'r')
        self._nx = int(hdf.root.input._v_attrs.nodex)
        self._ny = int(hdf.root.input._v_attrs.nodey)
        self._nz = int(hdf.root.input._v_attrs.nodez)
        
        #Clip against boundaries
        if nx_redu>=0 or nx_redu>=self._nx:
            nx_redu = self._nx-1
        if ny_redu==0 or ny_redu>=self._ny:
            ny_redu = self._ny-1
        if nz_redu==0 or nz_redu>=self._nz:
            nz_redu = self._nz-1
        
        
        self._nx_redu = nx_redu
        self._ny_redu = ny_redu
        self._nz_redu = nz_redu
        #Number of Timesteps in scene    \
        self.timesteps = int(hdf.root.time.nrows)
        #self._radius_inner = float(hdf.root.input._v_attrs.radius_inner) #Only important for displaying data. 
        #self._radius_outer = float(hdf.root.input._v_attrs.radius_outer)
        nproc_surf = int(hdf.root.input._v_attrs.nproc_surf)
        
        hexgrid = self.citcom2vtk(timestep,hdf,nproc_surf,nx_redu,ny_redu,nz_redu,surface,bottom)
        
        hdf.close()
        return hexgrid 
        
        