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
Cut-paste and combine Citcom surf data

usage: combinesurf.py modelname timestep nodex nodey nodez n_surf_proc nprocx nprocy nprocz
'''

class CombineSurf(object):


    def __init__(self, grid):
        # data storage
        self.saved = [''] * (grid['nox'] * grid['noy'])
        return



    def readData(self, crdfilename, surffilename, grid, cap):
        fp1 = file(crdfilename, 'r')
        fp2 = file(surffilename, 'r')

        # processor geometry
        nprocx = int(cap['nprocx'])
        nprocy = int(cap['nprocy'])
        nprocz = int(cap['nprocz'])

        # mesh geometry
        nox = grid['nox']
        noy = grid['noy']
        noz = grid['noz']

        mynox = 1 + (nox-1)/nprocx
        mynoy = 1 + (noy-1)/nprocy
        mynoz = 1 + (noz-1)/nprocz
        snodes = mynox * mynoy

        # skip header
        fp1.readline()
        fp2.readline()

        # read files and sanity check
        surf = fp2.readlines()
        if not (len(surf) == snodes):
            print '"%s" file size incorrect' % surffilename

        crd = fp1.readlines()
        if not (len(crd) == snodes*mynoz):
            print '"%s" file size incorrect' % crdfilename

        # paste data
        data = range(snodes)
        for i in range(len(data)):
            x, y, z = crd[(i+1)*mynoz-1].split()
            data[i] = '%s %s %s' % (x, y, surf[i])

        return data



    def join(self, data, me, grid, cap):
        # processor geometry
        nprocx = cap['nprocx']
        nprocy = cap['nprocy']
        nprocz = cap['nprocz']

        mylocz = me % nprocz
        mylocx = ((me - mylocz) / nprocz) % nprocx
        mylocy = (((me - mylocz) / nprocz - mylocx) / nprocx) % nprocy
        #print me, mylocx, mylocy, mylocz

        # mesh geometry
        nox = int(grid['nox'])
        noy = int(grid['noy'])
        noz = int(grid['noz'])

        mynox = 1 + (nox-1)/nprocx
        mynoy = 1 + (noy-1)/nprocy

        if not len(data) == mynox * mynoy:
            raise ValueError, "data size"

        mynxs = (mynox - 1) * mylocx
        mynys = (mynoy - 1) * mylocy

        n = 0
        for i in range(mynys, mynys+mynoy):
            for j in range(mynxs, mynxs + mynox):
                m = j + i * nox
                self.saved[m] = data[n]
                n += 1

        return



    def write(self, filename, grid, data):
        fp = file(filename, 'w')
        fp.write('%d x %d\n' % (grid['nox'], grid['noy']))
        fp.writelines(data)
        return



if __name__ == '__main__':

    import sys

    if not len(sys.argv) == 10:
        print __doc__
        sys.exit(1)

    prefix = sys.argv[1]
    step = int(sys.argv[2])

    grid = {}
    grid['nox'] = int(sys.argv[3])
    grid['noy'] = int(sys.argv[4])
    grid['noz'] = int(sys.argv[5])

    nprocxy = int(sys.argv[6])
    cap = {}
    cap['nprocx'] = int(sys.argv[7])
    cap['nprocy'] = int(sys.argv[8])
    cap['nprocz'] = int(sys.argv[9])

    nproc_per_cap = cap['nprocx'] * cap['nprocy'] * cap['nprocz']
    for i in range(nprocxy):
        cb = CombineSurf(grid)
        for n in range(i * nproc_per_cap + (cap['nprocz']-1), (i+1) * nproc_per_cap, cap['nprocz']):
            crdfilename = '%s.coord.%d' % (prefix, n)
            surffilename = '%s.surf.%d.%d' % (prefix, n, step)
            print 'reading', surffilename
            data = cb.readData(crdfilename, surffilename, grid, cap)
            cb.join(data, n, grid, cap)

        filename = '%s.surf%d.%d' % (prefix, i, step)
        print 'writing', filename
        cb.write(filename, grid, cb.saved)


# version
# $Id$

# End of file
