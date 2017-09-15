"""Creates points os a specified cap with a specified resolution.
"""

# Author: Martin Weier
#Copyright (C) 2006  California Institute of Technology
#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 2 of the License, or
#any later version.
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#You should have received a copy of the GNU General Public License
#along with this program; if not, write to the Free Software
#Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301 USA

from math import sqrt,acos,pi,atan, atan2, sin, cos


class Sphere(object):

    #Cointains corner points for each cap.
    lookup_table = [( (0.013497673906385899, 0.0, 0.54983437061309814) ,
                      (0.0077419690787792206, 0.4505336582660675, 0.31537202000617981) ,
                      (0.45813989639282227, 0.0, 0.30431538820266724) ,
                      (0.38879159092903137, 0.3889087438583374, -0.0095440549775958061) ),
                      
                    ( (0.0077419690787792206, 0.4505336582660675, 0.31537202000617981) ,
                      (-0.38879168033599854, 0.38890865445137024, 0.0095444004982709885) ,
                      (0.38879159092903137, 0.3889087438583374, -0.0095440549775958061) ,
                      (-0.007742113433778286, 0.45053350925445557, -0.31537219882011414) ),
                      
                    ( (-0.38879168033599854, 0.38890865445137024, 0.0095444004982709885) ,
                      (-0.45813983678817749, -1.4928090763532964e-07, -0.3043154776096344) ,
                      (-0.007742113433778286, 0.45053350925445557, -0.31537219882011414) ,
                      (-0.013497666455805302, -4.3980978858826347e-09, -0.54983437061309814) ),
                      
                    ( (0.013497673906385899, 0.0, 0.54983437061309814) ,
                      (-0.44265598058700562, -1.4423562788579147e-07, 0.32642868161201477) ,
                      (0.0077419690787792206, 0.4505336582660675, 0.31537202000617981) ,
                      (-0.38879168033599854, 0.38890865445137024, 0.0095444004982709885) ),
                      
                    ( (-0.44265598058700562, -1.4423562788579147e-07, 0.32642868161201477) ,
                      (-0.38879179954528809, -0.38890856504440308, 0.0095444004982709885) ,
                      (-0.38879168033599854, 0.38890865445137024, 0.0095444004982709885) ,
                      (-0.45813983678817749, -1.4928090763532964e-07, -0.3043154776096344) ),
                      
                    ( (-0.38879179954528809, -0.38890856504440308, 0.0095444004982709885) ,
                      (-0.0077417660504579544, -0.45053350925445557, -0.31537219882011414) ,
                      (-0.45813983678817749, -1.4928090763532964e-07, -0.3043154776096344) ,
                      (-0.013497666455805302, -4.3980978858826347e-09, -0.54983437061309814) ),
                      
                    ( (0.013497673906385899, 0.0, 0.54983437061309814) ,
                      (0.0077417795546352863, -0.4505336582660675, 0.31537202000617981) ,
                      (-0.44265598058700562, -1.4423562788579147e-07, 0.32642868161201477) ,
                      (-0.38879179954528809, -0.38890856504440308, 0.0095444004982709885) ),
                      
                    ( (0.0077417795546352863, -0.4505336582660675, 0.31537202000617981) ,
                      (0.38879171013832092, -0.38890862464904785, -0.0095440549775958061) ,
                      (-0.38879179954528809, -0.38890856504440308, 0.0095444004982709885) ,
                      (-0.0077417660504579544, -0.45053350925445557, -0.31537219882011414) ),
                      
                    ( (0.38879171013832092, -0.38890862464904785, -0.0095440549775958061) ,
                      (0.44265609979629517, -1.3367842655043205e-07, -0.32642853260040283) ,
                      (-0.0077417660504579544, -0.45053350925445557, -0.31537219882011414) ,
                      (-0.013497666455805302, -4.3980978858826347e-09, -0.54983437061309814) ),
                      
                    ( (0.013497673906385899, 0.0, 0.54983437061309814) ,
                      (0.45813989639282227, -1.3835439460763155e-07, 0.30431538820266724) ,
                      (0.0077417795546352863, -0.4505336582660675, 0.31537202000617981) ,
                      (0.38879171013832092, -0.38890862464904785, -0.0095440549775958061) ),
                      
                    ( (0.45813989639282227, -1.3835439460763155e-07, 0.30431538820266724) ,
                      (0.38879159092903137, 0.3889087438583374, -0.0095440549775958061) ,
                      (0.38879171013832092, -0.38890862464904785, -0.0095440549775958061) ,
                      (0.44265609979629517, -1.3367842655043205e-07, -0.32642853260040283) ),
                    
                    ( (0.38879159092903137, 0.3889087438583374, -0.0095440549775958061) ,
                      (-0.007742113433778286, 0.45053350925445557, -0.31537219882011414) ,
                      (0.44265609979629517, 0.0, -0.32642853260040283) ,
                      (-0.013497666455805302, -4.3980978858826347e-09, -0.54983437061309814) )]
        
    
    def cart2spherical(self,x,y,z):
        xypow = x**x+y**y
        r = sqrt(xypow+z**z)
        if y >= 0:
            phi = acos(x/sqrt(x/xypow))
        else:
            phi = 2*pi-acos(x/sqrt(xypow))
        theta = pi/2-atan(z/sqrt(xypow))
        return r,phi,theta
    
    
    def coords_of_cap(self,radius,resolution_x,resolution_y,cap):
        coords = []
        #radius = sqrt(c1x**2+c1y**2+c1z**2)
        
       
        c1x = self.lookup_table[cap][0][0]
        c1y = self.lookup_table[cap][0][1]
        c1z = self.lookup_table[cap][0][2]
            
        c2x = self.lookup_table[cap][1][0]
        c2y = self.lookup_table[cap][1][1]
        c2z = self.lookup_table[cap][1][2]
            
        c3x = self.lookup_table[cap][2][0]
        c3y = self.lookup_table[cap][2][1]
        c3z = self.lookup_table[cap][2][2]
            
        c4x = self.lookup_table[cap][3][0]
        c4y = self.lookup_table[cap][3][1]
        c4z = self.lookup_table[cap][3][2]
            
        coords1,theta,phi = self.evenly_divide_arc(resolution_x, c1x, c1y, c1z, c2x, c2y, c2z)
        coords2,theta,phi = self.evenly_divide_arc(resolution_x, c3x, c3y, c3z, c4x, c4y, c4z)
        
        
              
        for i in xrange(len(coords1)):
            temp,theta,phi = self.evenly_divide_arc(resolution_y, coords1[i][0],coords1[i][1],coords1[i][2], coords2[i][0],coords2[i][1],coords2[i][2])
            
            for j in xrange(len(theta)):
                x,y,z = self.RTF2XYZ(theta[j],phi[j],radius)
                coords.append((x,y,z))
        return coords

    def evenly_divide_arc(self,elx,x1,y1,z1,x2,y2,z2):
        nox=elx+1
        dx = (x2-x1)/elx
        dy = (y2-y1)/elx
        dz = (z2-z1)/elx
        theta = []
        phi = []
        coords = []
        for j in xrange(1,nox+1):
            x_temp = x1 + dx * (j-1) + 5.0e-32
            y_temp = y1 + dy * (j-1)
            z_temp = z1 + dz * (j-1)
            coords.append((x_temp,y_temp,z_temp))
            theta.append(acos(z_temp/sqrt(x_temp**2+y_temp**2+z_temp**2)))
            phi.append(self.myatan(y_temp,x_temp))
        return coords,theta,phi
    
    def RTF2XYZ(self,thet, phi, r):
        x = r * sin(thet) * cos(phi)
        y = r * sin(thet) * sin(phi)
        z = r * cos(thet)
        return x, y, z

    def myatan(self,y,x):
        fi = atan2(y,x)
        if fi<0.0:
            fi += 2*pi
        return fi


    
