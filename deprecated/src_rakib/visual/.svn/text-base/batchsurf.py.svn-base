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

'''
Combine Citcom Surface Data

usage: batchsurf.py machinefile modeldir modelname timestep nodex nodey nodez n_surf_proc nprocx nprocy nprocz
'''

if __name__ == '__main__':

    import sys, os

    if not len(sys.argv) == 12:
        print __doc__
        sys.exit(1)

    machinefile = sys.argv[1]
    modeldir = sys.argv[2]
    modelname = sys.argv[3]
    timestep = int(sys.argv[4])
    nodex = int(sys.argv[5])
    nodey = int(sys.argv[6])
    nodez = int(sys.argv[7])
    ncap = int(sys.argv[8])
    nprocx = int(sys.argv[9])
    nprocy = int(sys.argv[10])
    nprocz = int(sys.argv[11])

    # generate a string of machine names
    from batchcombine import machinefile2nodes
    totalnodes = nprocx * nprocy * nprocz * ncap
    nodes = machinefile2nodes(machinefile, totalnodes)

    # extract machines at the surface
    tmp = nodes.split()[(nprocz-1)::nprocz]
    nodes = ' '.join(tmp)

    # get coordinate, if necessary
    coord_exist = True
    for n in range(totalnodes):
        coord_file = '%(modelname)s.coord.%(n)d' % vars()
        coord_exist &= os.path.exists(coord_file)

    if not coord_exist:
        cmd = 'getcoord.sh %(modeldir)s %(modelname)s %(nprocz)d %(nodes)s' % vars()
        print cmd
        os.system(cmd)

    # get surf data
    cmd = 'getsurf.sh %(modeldir)s %(modelname)s %(timestep)d %(nprocz)d %(nodes)s' % vars()
    print cmd
    os.system(cmd)

    # combine
    cmd = 'combinesurf.py %(modelname)s %(timestep)d %(nodex)d %(nodey)d %(nodez)d %(ncap)d %(nprocx)d %(nprocy)d %(nprocz)d' % vars()
    print cmd
    os.system(cmd)

    # delete modelname.surf.* files, only if the originial data is not
    # stored in the current directory, ie. if velofile exists
    velofile = '%(modelname)s.velo.0.%(timestep)d' % vars()
    if not os.path.exists(velofile):
        cmd = 'rm %(modelname)s.surf.[0-9]*.%(timestep)d' % vars()
        print cmd
        os.system(cmd)

    # create .general file
    cmd = 'dxgeneralsurf.sh %(modelname)s.surf[0-9]*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)


# version
# $Id$

# End of file
