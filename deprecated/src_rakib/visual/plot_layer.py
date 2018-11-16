#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# plot_layer.py by Eh Tan.
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

"""Plotting CitcomS temperature at the specified layer

usage: plot_layer.py modelname caps step layer
  modelname: prefix of the combined capfile(s)
  caps: 1 (regional) or 12 (full) caps
  step: time step to plot
  layer: which layer to plot (0 to nodez-1)

input file:
  modelname.capXX.step - CitcomS combined cap file
  infile

output file:
  modelname.capXX.step.zYYY - extracted layer YYY from the cap file(s)
  modelname.capXX.step.zYYY.ps - Postscript image of layer YYY

"""


def read_params(infile):
    ## open input file and read header
    nodex, nodey, nodez = open(infile).readline().split('x')
    nodex = int(nodex)
    nodey = int(nodey)
    nodez = int(nodez)
    #print nodex, nodey, nodez
    return nodex, nodey, nodez


def find_minmax(zfile, nodex, nodey):
    fp = open(zfile)
    x = range(nodex*nodey)
    y = range(nodex*nodey)
    n = 0
    for line in fp.readlines():
        x[n] = float(line.split()[0])
        y[n] = float(line.split()[1])
        n = n + 1
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    fp.close()

    #print xmin, xmax, ymin, ymax
    return xmin, xmax, ymin, ymax


#######################################################################
# Main
#######################################################################
import sys, os

if len(sys.argv) != 5:
    print __doc__
    sys.exit(1)

modelname = sys.argv[1]
caps = int(sys.argv[2])
step = int(sys.argv[3])
layer = int(sys.argv[4])

## read model parameters
infile = '%s.cap%02d.%d' % (modelname, 0, step)
nodex, nodey, nodez = read_params(infile)


## slice the capfile(s), results saved in zfile
import zslice
all_zfiles = ''
for cap in range(caps):
    capfile = '%s.cap%02d.%d' % (modelname, cap, step)
    zfile = zslice.zslicefile(capfile, layer)
    all_zfiles = all_zfiles + ' ' + zfile
    if not os.path.exists(zfile):
        zslice.zslice(capfile, layer)


#######################################################################
# define GMT parameters
#######################################################################

## width of the plot
mapwidth = 6.0

if caps == 1:
    ## find min/max of the coordinate
    xmin, xmax, ymin, ymax = find_minmax(zfile, nodex, nodey)
    bounds = '%f/%f/%f/%f' % (xmin,xmax,ymin,ymax)
    proj = 'M%f' % mapwidth
    resolution = '%f/%f' % ( (xmax-xmin)/nodex, (ymax-ymin)/nodey )
else:
    ## map centered at Pacific
    bounds = '0/360/-90/90'
    proj = 'H180/%d' % mapwidth
    ## map centered at Greenwich
    #bounds = '-180/180/-90/90'
    #proj = 'H0/%d' % mapwidth
    resolution = '0.5'

cptfile = 'zz.cpt'
grdfile = '%s.%d.z%03d.tgrd' % (modelname, step, layer)
psfile = '%s.%d.z%03d.ps' % (modelname, step, layer)


#print 'Plotting...'

#######################################################################
## call GMT commands to do the followings:
## 1. cut the 1st, 2nd, 6th columns (lat, lon, temperature) of zfiles
## 2. using GMT's "surface" to generate the grdfile
## 3. generate color palette
## 4. plot the grdfile
## 5. plot the coastlines
## 6. set the font size for the colorbar
## 7. plot the colorbar
## 8. remove the grdfile and cptfile
#######################################################################

## min/max values to truncate temperature field
tmin = 0
tmax = 1

command = '''
cut -d' ' -f1,2,6 %(all_zfiles)s | \
    surface -I%(resolution)s -G%(grdfile)s -R%(bounds)s \
            -Ll%(tmin)d -Lu%(tmax)d

makecpt -Cpolar -T0/1/.1 > %(cptfile)s

grdimage %(grdfile)s -C%(cptfile)s -Bg90 -R%(bounds)s -J%(proj)s -X1 -Y2.0 -P -K > %(psfile)s

pscoast -R%(bounds)s -J%(proj)s -Bg90 -W -Dc -K -O >> %(psfile)s

gmtset  ANOT_FONT_SIZE 9

psscale -C%(cptfile)s -D8.5/2.25/4.0/0.25 -O >> %(psfile)s

rm -f %(grdfile)s %(cptfile)s
''' % vars()

#print command
os.system(command)

#print 'Done'


# version
# $Id: plot_layer.py 6402 2007-03-26 18:11:02Z tan2 $

# End of file
