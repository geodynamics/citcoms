#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
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
        self.saved = range(grid['nox'] * grid['noy'])
        return



    def readData(self, crdfilename, surffilename, grid):
        fp1 = file(crdfilename, 'r')
        fp2 = file(surffilename, 'r')

        # mesh geometry
        nox = grid['nox']
        noy = grid['noy']
        noz = grid['noz']
        snodes = nox*noy

        # skip header
        fp1.readline()
        fp2.readline()

        # read files and sanity check
        surf = fp2.readlines()
        if not (len(surf) == snodes):
            print '"%s" file size incorrect' % surffilename

        crd = fp1.readlines()
        if not (len(crd) == snodes*noz):
            print '"%s" file size incorrect' % crdfilename

        # paste data
        data = range(snodes)
        for i in range(len(data)):
            x, y, z = crd[(i+1)*noz-1].split()
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
        nox = grid['nox']
        noy = grid['noy']
        noz = grid['noz']

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
        n  = (i+1) * nproc_per_cap - 1
        crdfilename = '%s.coord.%d' % (prefix, n)
        surffilename = '%s.surf.%d.%d' % (prefix, n, step)
        print 'reading', surffilename
        data = cb.readData(crdfilename, surffilename, grid)
        cb.join(data, n, grid, cap)

        filename = '%s.surf%d.%d' % (prefix, i, step)
        print 'writing', filename
        cb.write(filename, grid, cb.saved)


# version
# $Id: combinesurf.py,v 1.1 2004/06/08 01:35:06 tan2 Exp $

# End of file
