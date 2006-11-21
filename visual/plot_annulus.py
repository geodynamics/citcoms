#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# plot_annulus.py by Eh Tan.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"""Plot an annulus cross-section from full spherical CitcomS output

usage: plot_annulus.py modelname step
  modelname: prefix of CitcomS datafile
  step: time step to plot

input:
  modelname.capXX.step

output:
  modelname.step.z###.tgrd
  modelname.xsect.step.ps
"""


def get_radius(modelname, step):
    ### get nodez ###
    capfile = get_capfile(modelname, 0, step)
    data = open(capfile)
    header = data.readline()
    nodex, nodey, nodez = header.split('x')
    nodez = int(nodez)

    ### read z coordinate ###
    radius = range(nodez)
    for i in range(nodez):
        radius[i] = float(data.readline().split()[2])

    data.close()

    return radius


def get_capfile(modelname, cap, step):
    return '%s.cap%02d.%d' % (modelname, cap, step)


def great_circle_proj():
    from math import pi, sin, cos, tan, atan
    while 1:
        text = '''Choose method to specify great circle path:
          (1) a center and an azimuth
          (2) a center and another point on the path
          (3) a center and a rotation pole position\n'''
        option = int(raw_input(text))

        if 1 <= option <= 3:
            lon = float(raw_input('Center longitude: '))
            lat = float(raw_input('Center latitude: '))
            center = '-C%f/%f -L-180/180' % (lon, lat)
            if option == 1:
                az = float(raw_input('Azimuth (in degrees clockwise from north): '))
                proj = '%s -A%f' % (center, az)
                break

            if option == 2:
                elon = float(raw_input('2nd point longitude: '))
                elat = float(raw_input('2nd point latitude: '))
                r2d = 180.0 / pi
                ## transfer to azimuth mode
                b = (90 - elat) / r2d
                a = (90 - lat) / r2d
                delta = (elon - lon) / r2d
                if abs(lat) == 90:
                    ## on the pole
                    print 'pole cannot be the center.'
                    continue
                elif (elon - lon) % 180 == 0:
                    ## on the same meridian
                    az = 0
                elif lat == 0 and elat == 0:
                    ## on the equator
                    az = 90
                else:
                    az = atan((sin(a)/tan(b) - cos(a)*cos(delta)) / sin(delta))
                    az = 90 - r2d * az

                proj = '%s -A%f' % (center, az)
                break

            if option == 3:
                lon = float(raw_input('Pole longitude: '))
                lat = float(raw_input('Pole latitude: '))
                proj = '%s -T%f/%f' % (center, lon, lat)
                break

        else:
            print 'Incorrect mode!\n'
            continue

    return proj


#######################################################################
# Main
#######################################################################

import os, sys
import zslice

if len(sys.argv) != 3:
    print __doc__
    sys.exit(0)

modelname = sys.argv[1]
step = int(sys.argv[2])


### get radius ###
radius = get_radius(modelname, step)
nodez = len(radius)
#print radius


### slice the capfile for all layers ###
for cap in range(12):
    capfile = get_capfile(modelname, cap, step)
    exist = 1
    for layer in range(nodez):
        zfile = zslice.zslicefile(capfile, layer)
        exist = exist and os.path.isfile(zfile)

    if not exist:
        #print 'Creating zslice files'
        zslice.zslice(capfile)


### create great circle path ###
gcproj = great_circle_proj()
gcfile = 'circle.xyp'
az_resolution = 0.5
command = 'project %(gcproj)s -G%(az_resolution)f > %(gcfile)s' % vars()
os.system(command)


### create cross section along the great circle ###

## range of layers to plot
botlayer = 0
toplayer = nodez - 1

bounds = '0/360/-90/90'
xsectfile = '%s.xsect.%d.xyz' % (modelname, step)
out = open(xsectfile,'w')

for layer in range(botlayer, toplayer+1):
    ## gather the filenames of all zfiles
    all_zfiles = ''
    for cap in range(12):
        capfile = get_capfile(modelname, cap, step)
        zfile = zslice.zslicefile(capfile, layer)
        all_zfiles = all_zfiles + ' ' + zfile

    ## create a grdfile for each layer
    grdfile = '%s.%d.z%03d.tgrd' % (modelname, step, layer)
    if not os.path.isfile(grdfile):
        command = '''
cut -d' ' -f1,2,6 %(all_zfiles)s | \
    surface -I%(az_resolution)s -G%(grdfile)s -R%(bounds)s -N1
''' % vars()
        os.system(command)

    ## sampling the grdfile along the great circle
    xyptfp = os.popen('grdtrack %(gcfile)s -G%(grdfile)s -Lg' % vars())

    ## write the sampled results (azimuth, r, temperature) to a xect file
    for line in xyptfp.readlines():
        xypt = line.split()
        out.write('%s\t%f\t%s\n' % (xypt[2], radius[layer], xypt[3]) )
    xyptfp.close()

out.close()


### Plotting ###
#print 'Plotting'
psfile = '%s.xsect.%d.ps' % (modelname, step)
mapwidth = 12.0
proj = 'H0/%f' % mapwidth
yshift = mapwidth * 1.5
cptfile = 'zz.cpt'

## colorbar length and location
cbarh = mapwidth / 2
cbxshift = mapwidth + .5
cbyshift = cbarh / 2

## plot the temperature field at mid-depth and a great circle
grdfile = '%s.%d.z%03d.tgrd' % (modelname, step, int(nodez/2))
command = '''
makecpt -Cpolar -T0/1/.1 > %(cptfile)s

grdimage %(grdfile)s -C%(cptfile)s -Bg360 -R%(bounds)s -J%(proj)s -X1.5 -Y%(yshift)f -P -K > %(psfile)s

pscoast -R%(bounds)s -J%(proj)s -W -Dc -K -O >> %(psfile)s

psxy %(gcfile)s -R%(bounds)s -J%(proj)s -W6. -O -K >> %(psfile)s

gmtset  ANOT_FONT_SIZE 9
psscale -C%(cptfile)s -D%(cbxshift)f/%(cbyshift)f/%(cbarh)f/0.25 -K -O >> %(psfile)s
''' % vars()
os.system(command)


## create a polar coordinate plot of the xsection
##
## TODO: there is always a gap on the left side of the annulus. How to fix it?
grdfile2 = 'xsection.grd'
bounds2 = '-180/180/%f/%f' % (radius[botlayer], radius[toplayer])
r_resolution = (radius[toplayer] - radius[botlayer]) / 100
resolution = '%f/%f' % (az_resolution, r_resolution)
yshift = mapwidth * 1.2

command = '''
surface %(xsectfile)s -G%(grdfile2)s -I%(resolution)s -R%(bounds2)s

grdimage %(grdfile2)s -C%(cptfile)s -JP%(mapwidth)fa -B30ns -R%(bounds2)s -X0.2 -Y-%(yshift)f -P -O >> %(psfile)s

rm -f label.txt %(cptfile)s %(gcfile)s %(xsectfile)s %(grdfile2)s
''' % vars()
os.system(command)


# version
# $Id$

# End of file
