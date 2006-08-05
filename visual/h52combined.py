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

'''Convert the HDF5 output file to ASCII file(s), with the same format as the
combined file

usage: h52combined.py h5file step1 [step2 [...] ]
'''

import tables


def find_prefix(h5file):
    suffix = '.h5'

    if h5file.endswith(suffix):
        prefix = h5file[:-len(suffix)]
    else:
        prefix = h5file

    return prefix



def convert(h5file, prefix, record):
    print 'in convert():', h5file, prefix, record

    f = tables.openFile(h5file)

    try:
        # loop through all the caps
        for cap in range(12):
            cap_no = 'cap%02d' % cap

            # get 'cap_no' group, if no such group, return None
            group = _get_hdf_group(f.root, cap_no)
            print repr(group)
            if group is None:
                break

            x = group.coord
            v = group.velocity[record,:]
            t = group.temperature[record,:]
            visc = group.viscosity[record,:]

            # TODO: map record -> step
            outputfile = '%s.%s.%d' % (prefix, cap_no, record)
            #print outputfile

            output(outputfile, x, v, t, visc)



    finally:
        f.close()

    return outputfile



def output(outputfile, x, v, t, visc):
    out = file(outputfile, 'w')
    try:
        # write header (shape of the arrays)
        nx, ny, nz = t.shape[:3]
        header = '%d x %d x %d\n' % (nx, ny, nz)
        out.write(header)

        # write data
        for i in range(ny):
            for j in range(nx):
                for k in range(nz):
                    #n = k + j*nz + i*nz*nx
                    xx = x[j, i, k, :]
                    vv = v[j, i, k, :]
                    tt = t[j, i, k]
                    hh = visc[j, i, k]
                    format = '%.6e '*7 + '%.6e\n'
                    line = format % (
                        xx[0], xx[1], xx[2],
                        vv[0], vv[1], vv[2],
                        tt, hh )
                    out.write(line)

    finally:
        out.close()



def _get_hdf_group(base, child):
    try:
        return base._f_getChild(child)
    except tables.exceptions.NoSuchNodeError:
        return None



def make_general(outputfile):
    import os

    path = os.path.dirname(__file__)
    cmd = '%s/dxgeneral.sh %s' % (path, outputfile)
    #print cmd
    os.system(cmd)



if __name__ == '__main__':
    import sys

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(1)


    h5file = sys.argv[1]
    file_prefix = find_prefix(h5file)

    steps = [ int(x) for x in sys.argv[2:] ]


    for step in steps:
        # write to outputfile
        outputfile = convert(h5file, file_prefix, step)

        # generate header file for OpenDX
        make_general(outputfile)

