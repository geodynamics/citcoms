#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""
Slice horizontal layer(s) from CitcomS output (from batchcombine.py).
Also convert the (x, y) coordinate to (lon, lat).

usage:
  zslice(prefix, layer1 [,layer2 [,...]] )
    -- slice the specified layer(s)
       layer# : layer(s) to plot (count from 0)

  zslice(prefix)
    -- slice all layers

  It can be used in the command line as well.

input:
  prefix
outout:
  prefix.z###
"""

all = ('zslice', 'zslicefile')


def zslicefile(prefix, layer):
    return '%s.z%03d' % (prefix, layer)


def zslice(prefix, *ilayers):

    from math import pi
    r2d = 180.0 / pi

    ## open cap file and read header
    capfile = open(prefix)
    try:
        optfile = open(prefix.replace('cap', 'opt'))
    except:
        optfile = None
    nodex, nodey, nodez = capfile.readline().split('x')
    nodez = int(nodez)
    #print nodez, ilayers

    ## validate ilayers
    layers = check_layers(ilayers, nodez)
    nlayer = len(layers)

    ## read opt file header
    if optfile is not None:
        optfile.readline()

    ## allocate arrays
    output = range(nlayer)
    lines = range(nodez)
    lines2 = range(nodez)

    ## open output files
    for i in range(nlayer):
        zfile = zslicefile(prefix, layers[i])
        output[i] = open(zfile, 'w')

    try:
        while 1:
            ## read nodez lines
            for j in range(nodez):
                lines[j] = capfile.readline().strip()

                if optfile is not None:
                    lines2[j] = optfile.readline().strip()
                else:
                    lines2[j] = ''

            ## file is empty or EOF?
            if not lines[nodez-1]:
                break

            for i in range(nlayer):
                layer = layers[i]

                ## spilt the first 3 columns only, the rest will be
                ## output as-is
                data = lines[layer].split(' ', 3)
                data2 = lines2[layer]
                lat = 90 - float(data[0])*r2d
                lon = float(data[1])*r2d
                output[i].write( '%f %f %s %s\n' % (lon, lat, data[3], data2) )

        capfile.close()
        if optfile is not None:
            optfile.close()

    finally:
        for i in range(nlayer):
            output[i].close()


    return



def check_layers(layers, nodez):
    if layers == ():
        ## if empty, we will slice every layer
        layers = range(nodez)
    else:
        ## otherwise, check bounds of layers
        for layer in layers:
            if not (0<= layer < nodez):
                raise ValueError, 'layer out of range (0-%d)' % (nodez - 1)

    return layers



## if run as a script
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(1)

    prefix = sys.argv[1]
    layers = [ int(x) for x in sys.argv[2:] ]

    zslice(prefix, *layers)


# version
# $Id: zslice.py 13257 2008-11-04 21:02:08Z tan2 $

# End of file
