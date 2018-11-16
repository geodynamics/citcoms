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


    def __init__(self, nodex, nodey, nodez, nprocx, nprocy, nprocz):
        self.nodex = nodex
        self.nodey = nodey
        self.nodez = nodez

        self.nprocx = nprocx
        self.nprocy = nprocy
        self.nprocz = nprocz

        # data storage
        self.saved = range(nodex * nodey * nodez)
        return



    def readData(self, filename, headlines=1):
        fp = file(filename, 'r')

        # discard the header
        for i in range(headlines):
            header = fp.readline()
            #print header

        return fp.readlines()



    def join(self, data, me):
        # processor geometry
        nprocx = self.nprocx
        nprocy = self.nprocy
        nprocz = self.nprocz

        mylocz = me % nprocz
        mylocx = ((me - mylocz) / nprocz) % nprocx
        mylocy = (((me - mylocz) / nprocz - mylocx) / nprocx) % nprocy
        #print me, mylocx, mylocy, mylocz

        # mesh geometry
        nodex = self.nodex
        nodey = self.nodey
        nodez = self.nodez

        mynodex = 1 + (nodex-1)/nprocx
        mynodey = 1 + (nodey-1)/nprocy
        mynodez = 1 + (nodez-1)/nprocz

        if not len(data) == mynodex * mynodey * mynodez:
            raise ValueError, "incorrect data size"

        mynxs = (mynodex - 1) * mylocx
        mynys = (mynodey - 1) * mylocy
        mynzs = (mynodez - 1) * mylocz

        n = 0
        for i in range(mynys, mynys+mynodey):
            for j in range(mynxs, mynxs + mynodex):
                for k in range(mynzs, mynzs + mynodez):
                    m = k + j * nodez + i * nodex * nodez
                    self.saved[m] = data[n]
                    n += 1

        return



    def write(self, filename):
        fp = file(filename, 'w')
        header = '%d x %d x %d\n' % (self.nodex, self.nodey, self.nodez)
        #print header
        fp.write(header)
	fp.writelines(self.saved)
        return



##############################################


def combine(prefix, opts, step, nodex, nodey, nodez,
            ncap, nprocx, nprocy, nprocz):
    combined_files = []
    nproc_per_cap = nprocx * nprocy * nprocz
    for i in range(ncap):
        cb = Combine(nodex, nodey, nodez, nprocx, nprocy, nprocz)
        for n in range(i * nproc_per_cap, (i+1) * nproc_per_cap):
            filename = '%s.%s.%d.%d.pasted' % (prefix, opts, n, step)
            print 'reading', filename
            data = cb.readData(filename, 0)
            cb.join(data, n)

        if opts == 'coord,velo,visc':
            filename = '%s.cap%02d.%d' % (prefix, i, step)
        else:
            filename = '%s.opt%02d.%d' % (prefix, i, step)

        print 'writing', filename
        cb.write(filename)
        combined_files.append(filename)

    return combined_files


if __name__ == '__main__':

    import sys

    if not len(sys.argv) == 10:
        print __doc__
        sys.exit(1)

    prefix = sys.argv[1]
    step = int(sys.argv[2])

    nodex = int(sys.argv[3])
    nodey = int(sys.argv[4])
    nodez = int(sys.argv[5])

    ncap = int(sys.argv[6])
    nprocx = int(sys.argv[7])
    nprocy = int(sys.argv[8])
    nprocz = int(sys.argv[9])

    combine(prefix, step, nodex, nodey, nodez,
            ncap, nprocx, nprocy, nprocz)


# version
# $Id: combine.py 7748 2007-07-26 00:13:39Z tan2 $

# End of file
