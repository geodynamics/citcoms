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
Paste and combine Citcom data

Usage: batchcombine.py <machinefile | node-list> datadir datafile timestep nodex nodey nodez ncap nprocx nprocy nprocz
'''



def machinefile2nodes(machinefile, totalnodes):

    try:
        nodelist = file(machinefile).readlines()
    except IOError:
        nodelist = machinefile.split()

    # check the length of nodelist
    n = len(nodelist)
    if not n == totalnodes:
        print 'WARNING: length of machinefile does not match number of processors, try to duplicate machinefile...'
        if (totalnodes > n) and ((totalnodes % n) == 0):
            # try to match number of processors by duplicating nodelist
            nodelist *= (totalnodes / n)
        else:
            raise ValueError, 'incorrect machinefile size'

    # generate a string of machine names
    nodes = ''
    for node in nodelist:
        nodes += '%s ' % node.strip()

    return nodes



def combine(nodes, datadir, datafile, timestep, nodex, nodey, nodez,
            ncap, nprocx, nprocy, nprocz):
    import os

    # paste
    cmd = 'batchpaste.sh %(datadir)s %(datafile)s %(timestep)d %(nodes)s' \
          % vars()
    print cmd
    os.system(cmd)

    # combine
    cmd = 'combine.py %(datafile)s %(timestep)d %(nodex)d %(nodey)d %(nodez)d %(ncap)d %(nprocx)d %(nprocy)d %(nprocz)d' % vars()
    print cmd
    os.system(cmd)

    # delete
    cmd = 'rm %(datafile)s.[0-9]*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)

    # create .general file
    cmd = 'dxgeneral.sh %(datafile)s.cap*.%(timestep)d' % vars()
    print cmd
    os.system(cmd)

    return


if __name__ == '__main__':

    import sys

    if not len(sys.argv) == 12:
        print __doc__
        sys.exit(1)

    machinefile = sys.argv[1]
    datadir = sys.argv[2]
    datafile = sys.argv[3]
    timestep = int(sys.argv[4])
    nodex = int(sys.argv[5])
    nodey = int(sys.argv[6])
    nodez = int(sys.argv[7])
    ncap = int(sys.argv[8])
    nprocx = int(sys.argv[9])
    nprocy = int(sys.argv[10])
    nprocz = int(sys.argv[11])

    totalnodes = nprocx * nprocy * nprocz * ncap
    nodelist = machinefile2nodes(machinefile, totalnodes)

    combine(nodelist, datadir, datafile, timestep, nodex, nodey, nodez,
            ncap, nprocx, nprocy, nprocz)



# version
# $Id$

# End of file
