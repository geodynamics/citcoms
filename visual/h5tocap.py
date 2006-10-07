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

usage: h5tocap.py h5file step1 [step2 [...] ]
'''

import sys

try:
    import tables
except ImportError, exc:
    print "Import Error:", exc
    print """
This script needs the 'PyTables' extension package.
Please install the package before running the script, or you can use the
'h5tocap' program."""
    sys.exit(1)


def find_prefix(h5file):
    suffix = '.h5'

    if h5file.endswith(suffix):
        prefix = h5file[:-len(suffix)]
    else:
        prefix = h5file

    return prefix



def convert(h5fid, prefix, step):
    root = h5fid.root

    # get attribute 'caps' in the input group
    caps = int(root.input._v_attrs.caps)
    frame = get_frame(root, step)

    # loop through all the caps
    for cap in range(caps):
        cap_no = 'cap%02d' % cap

        x = root.coord[cap, :]
        v = root.velocity[frame, cap, :]
        t = root.temperature[frame, cap, :]
        visc = root.viscosity[frame, cap, :]

        outputfile = '%s.%s.%d' % (prefix, cap_no, step)

        print 'writing to', outputfile, '...'
        output(outputfile, x, v, t, visc)

    return



def get_frame(root, step):
    steps = list(root.time.col('step'))
    try:
        return steps.index(step)
    except ValueError:
        raise ValueError("step %d is not in the dataset" % step)



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


    h5file = sys.argv[1]
    # remove the pathname and '.h5' suffix, if present
    file_prefix = find_prefix(os.path.basename(h5file))

    steps = [ int(x) for x in sys.argv[2:] ]

    fid = tables.openFile(h5file)
    try:
        for step in steps:
            try:
                convert(fid, file_prefix, step)
            except ValueError, exc:
                print "Error: ", exc
    finally:
        fid.close()


# version
# $Id$

# End of file
