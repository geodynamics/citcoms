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

Usage: batchcombine.py <machinefile | node-list> datadir datafile timestep nodex nodey nodez ncap nprocx nprocy nprocz [ncompositions]
'''



def machinefile2nodes(machinefile, totalnodes):
    nodelist = machinefile2nodelist(machinefile, totalnodes)
    nodes = nodelist2nodes(nodelist)
    return nodes



def machinefile2nodelist(machinefile, totalnodes):
    '''Read the machinefile to get a list of machine names. If machinefile
    is not readable, treat it as a string containing the machine names.
    Return the list of machine names. The length of the list is totalnodes.
    '''
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

    return nodelist



def nodelist2nodes(nodelist):
    # generate a string of machine names
    nodes = ' '.join([x.strip() for x in nodelist])

    return nodes



def batchpaste(datadir, datafile, opts, timestep, nodes):
    from socket import gethostname
    hostname = gethostname()

    import os
    cwd = os.getcwd()

    for rank, node in enumerate(nodes):
        if node == 'localhost' or node == hostname:
            # local paste
            import pasteCitcomData
            pasteCitcomData.run(datadir, datafile, opts, rank, timestep, cwd)

        else:
            # remote paste

            # replace 'rsh' with 'ssh' if necessary
            remote_shell = 'rsh'

            cmd = '%(remote_shell)s %(node)s pasteCitcomData.py %(datadir)s %(datafile)s %(opts)s %(rank)d %(timestep)d %(cwd)s' % vars()
            os.system(cmd)

    return



def batchcombine(nodes, datadir, datafile, timestep, nodex, nodey, nodez,
                 ncap, nprocx, nprocy, nprocz, optional_fields,
                 ncompositions=0):
    # paste
    opts0 = 'coord,velo,visc'
    opts1 = optional_fields

    batchpaste(datadir, datafile, opts0, timestep, nodes)
    if opts1: batchpaste(datadir, datafile, opts1, timestep, nodes)

    # combine
    import combine
    combine.combine(datafile, opts0, timestep, nodex, nodey, nodez,
                        ncap, nprocx, nprocy, nprocz)
    if opts1: combine.combine(datafile, opts1, timestep, nodex, nodey, nodez,
                    ncap, nprocx, nprocy, nprocz)

    # delete pasted files
    import glob
    filenames = glob.glob('%(datafile)s.*.%(timestep)d.pasted' % vars())

    import os
    for filename in filenames:
        os.remove(filename)


    # create .general file
    import dxgeneral
    combined_files0 = []
    combined_files1 = []
    for cap in range(ncap):
        combined_files0.append('%(datafile)s.cap%(cap)02d.%(timestep)d' % vars())
        if opts1: combined_files1.append('%(datafile)s.opt%(cap)02d.%(timestep)d' % vars())

    dxgeneral.write(opts0, 0, combined_files0)
    if opts1: dxgeneral.write(opts1, ncompositions, combined_files1)

    return



if __name__ == '__main__':

    import sys

    if len(sys.argv) < 12:
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

    if len(sys.argv) < 13:
        ncompositions = 0
    else:
        ncompositions = int(sys.argv[12])

    totalnodes = nprocx * nprocy * nprocz * ncap
    nodelist = machinefile2nodelist(machinefile, totalnodes)

    batchcombine(nodelist, datadir, datafile, timestep, nodex, nodey, nodez,
                 ncap, nprocx, nprocy, nprocz, ncompositions)



# version
# $Id$

# End of file
