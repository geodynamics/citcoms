#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''
Combine the pasted Citcom Data

usage: combine.py modelname timestep nodex nodey nodez n_surf_proc nprocx nprocy nprocz
'''

class Combine(object):


    def __init__(self, grid):
        # data storage
        self.saved = range(grid['nox'] * grid['noy'] * grid['noz'])
        return



    def readData(self, filename):
        fp = file(filename, 'r')
        # header
        fp.readline()
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



    def write(self, filename, grid, data):
        fp = file(filename, 'w')
        fp.write('%d x %d x %d\n' % (grid['nox'], grid['noy'], grid['noz']))
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
        cb = Combine(grid)
        for n in range(i * nproc_per_cap, (i+1) * nproc_per_cap):
            filename = '%s.%d.%d' % (prefix, n, step)
            print 'reading', filename
            data = cb.readData(filename)
            cb.join(data, n, grid, cap)

        filename = '%s.cap%d.%d' % (prefix, i, step)
        print 'writing', filename
        cb.write(filename, grid, cb.saved)


# version
# $Id: combine.py,v 1.1 2004/01/14 23:22:52 tan2 Exp $

# End of file
