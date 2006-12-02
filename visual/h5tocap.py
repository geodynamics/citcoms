#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan
# Copyright (C) 2002-2006, California Institute of Technology.
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

'''
Convert the CitcomS HDF5 output file to ASCII file(s), with the format of the
combined cap files

usage: h5tocap.py modelname step1 [step2 [...] ]
'''

import sys


### check whether 'PyTables' is importable
try:
    import tables
except ImportError, exc:
    print "Import Error:", exc
    print """
This script needs the 'PyTables' extension package.
Please install the package before running the script, or you can use the
'h5tocap' program."""
    sys.exit(1)


def convert(h5fid, prefix, step):
    # this file contains time-independent data, e.g., coordinates
    root = h5fid.root

    # get attribute 'caps' (# of caps) in the input group
    caps = int(root.input._v_attrs.caps)

    # this file contains time-depedent data, e.g., velocity, temperature
    h5file2 = '%s.%d.h5' % (prefix, step)
    fid2 = tables.openFile(h5file2, 'r')
    root2 = fid2.root

    try:
        # loop through all the caps
        for cap in range(caps):
            x = root.coord[cap, :]
            v = root2.velocity[cap, :]
            t = root2.temperature[cap, :]
            visc = root2.viscosity[cap, :]

            outputfile = '%s.cap%02d.%d' % (prefix, cap, step)

            print 'writing to', outputfile, '...'
            output(outputfile, x, v, t, visc)
    finally:
        fid2.close()

    return



def output(outputfile, x, v, t, visc):
    out = file(outputfile, 'w')
    try:
        # write header (shape of the arrays)
        nx, ny, nz = t.shape[:3]
        header = '%d x %d x %d\n' % (nx, ny, nz)
        out.write(header)

        # write data
        format = '%.6e '*7 + '%.6e\n'
        for j in range(ny):
            for i in range(nx):
                for k in range(nz):
                    #n = k + i*nz + j*nz*nx
                    xx = x[i, j, k, :]
                    vv = v[i, j, k, :]
                    tt = t[i, j, k]
                    hh = visc[i, j, k]
                    line = format % (
                        xx[0], xx[1], xx[2],
                        vv[0], vv[1], vv[2],
                        tt, hh )
                    out.write(line)

    finally:
        out.close()

    return



if __name__ == '__main__':
    import sys, os.path

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(1)


    modelname = sys.argv[1]
    steps = [ int(x) for x in sys.argv[2:] ]

    h5file = modelname + '.h5'
    fid = tables.openFile(h5file, 'r')
    try:
        for step in steps:
            try:
                convert(fid, modelname, step)
            except ValueError, exc:
                print "Error: ", exc
    finally:
        fid.close()


# version
# $Id$

# End of file
