#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''
Combine the pasted Citcom Data

usage: combine.py modelname timestep nodex nodey nodez ncap nprocx nprocy nprocz
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



    def write(self, filename, grid):
        fp = file(filename, 'w')
        fp.write('%d x %d x %d\n' % (grid['nox'], grid['noy'], grid['noz']))
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
# $Id: combine.py,v 1.3 2005/06/10 02:23:25 leif Exp $

# End of file
