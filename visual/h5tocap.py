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

'''Convert the HDF5 output file to ASCII file(s), with the format of the
combined cap files

usage: h5tocap.py h5file frame1 [frame2 [...] ]
'''

import tables


def find_prefix(h5file):
    suffix = '.h5'

    if h5file.endswith(suffix):
        prefix = h5file[:-len(suffix)]
    else:
        prefix = h5file

    return prefix



def convert(h5file, prefix, frame):
    f = tables.openFile(h5file)
    try:
        # get attribute 'caps' in the input group
        caps = int(f.root.input._v_attrs.caps)
        steps = f.root.time.col('step')

        # loop through all the caps
        for cap in range(caps):
            cap_no = 'cap%02d' % cap
            group = getattr(f.root, cap_no)

            x = group.coord
            v = group.velocity[frame,:]
            t = group.temperature[frame,:]
            visc = group.viscosity[frame,:]

            outputfile = '%s.%s.%d' % (prefix, cap_no, steps[frame])

            print 'writing to', outputfile, '...'
            output(outputfile, x, v, t, visc)

    finally:
        f.close()

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
    import sys

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(1)


    h5file = sys.argv[1]
    file_prefix = find_prefix(h5file)

    frames = [ int(x) for x in sys.argv[2:] ]


    for frame in frames:
        # write to outputfile
        convert(h5file, file_prefix, frame)


