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
Combine the pasted Citcom Data

usage: combine.py datafile timestep nodex nodey nodez ncap nprocx nprocy nprocz
'''

class Combine(object):


    def __init__(self, grid):
        # data storage
        self.saved = range(grid['nox'] * grid['noy'] * grid['noz'])
        return



    def readData(self, filename):
        fp = file(filename, 'r')
        header = fp.readline()
        #print header
        return fp.readlines()



    def join(self, data, me, grid, cap):
        # processor geometry
        nprocx = int(cap['nprocx'])
        nprocy = int(cap['nprocy'])
        nprocz = int(cap['nprocz'])

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
        mynoz = 1 + (noz-1)/nprocz

        if not len(data) == mynox * mynoy * mynoz:
            raise ValueError, "data size"

        mynxs = (mynox - 1) * mylocx
        mynys = (mynoy - 1) * mylocy
        mynzs = (mynoz - 1) * mylocz

        n = 0
        for i in range(mynys, mynys+mynoy):
            for j in range(mynxs, mynxs + mynox):
                for k in range(mynzs, mynzs + mynoz):
                    m = k + j * noz + i * nox * noz
                    self.saved[m] = data[n]
                    n += 1

        return



    def write(self, filename, grid):
        fp = file(filename, 'w')
        header = '%d x %d x %d\n' % (grid['nox'], grid['noy'], grid['noz'])
        #print header
        fp.write(header)
	fp.writelines(self.saved)
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

    ncap = int(sys.argv[6])
    cap = {}
    cap['nprocx'] = int(sys.argv[7])
    cap['nprocy'] = int(sys.argv[8])
    cap['nprocz'] = int(sys.argv[9])

    nproc_per_cap = cap['nprocx'] * cap['nprocy'] * cap['nprocz']
    for i in range(ncap):
        cb = Combine(grid)
        for n in range(i * nproc_per_cap, (i+1) * nproc_per_cap):
            filename = '%s.%d.%d' % (prefix, n, step)
            print 'reading', filename
            data = cb.readData(filename)
            cb.join(data, n, grid, cap)

        filename = '%s.cap%d.%d' % (prefix, i, step)
        print 'writing', filename
        cb.write(filename, grid)


# version
# $Id$

# End of file
