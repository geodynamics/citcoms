#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

'''Create OpenDX .general file for combined Citcom Data

  Usage: dxgeneral.py combined_fields file1 [file2 [...]]
'''

import os, sys


def write(opts, filenames):

    for filename in filenames:
        if not os.path.exists(filename):
            print 'file "%s" does not exist' % filename
            sys.exit(1)

        shape = get_shape(filename)

        outfile = filename + '.general'
        f = open(outfile, 'w')

        try:
            write_general_file(f, filename, opts, shape)
        finally:
            f.close()

    return



def get_shape(filename):

    # the first line of the file contains the shape information
    header = open(filename).readline()
    shape = tuple([int(x) for x in header.split('x')])

    return shape



def write_general_file(f, filename, opts, shape):

    template = '''file = %(filename)s
grid = %(shape_str)s
format = ascii
interleaving = field
majority = row
header = lines 1
field = %(opt_field_str)s
structure = %(opt_struct_str)s
type = %(opt_type_str)s

end
'''

    # mapping from opt name to field name
    field = {'comp_nd': 'composition',
             'coord': 'locations',
             'pressure': 'pressure',
             'stress': 'stress',
             'velo': 'velocity, temperature',
             'visc': 'viscosity'}

    # mapping from opt name to data structure
    struct = {'comp_nd': 'scalar',
              'coord': '3-vector',
              'pressure': 'scalar',
              'stress': '6-vector',
              'velo': '3-vector, scalar',
              'visc': 'scalar'}

    # mapping from opt name to data type
    type = {'comp_nd': 'float',
            'coord': 'float',
            'pressure': 'float',
            'stress': 'float',
            'velo': 'float, float',
            'visc': 'float'}

    opt_field = []
    opt_struct = []
    opt_type = []
    for opt in opts.split(','):
        opt_field.append(field[opt])
        opt_struct.append(struct[opt])
        opt_type.append(type[opt])

    shape_str = ' x '.join([str(x) for x in shape])
    opt_field_str = ', '.join(opt_field)
    opt_struct_str = ', '.join(opt_struct)
    opt_type_str = ', '.join(opt_type)

    f.write(template % vars())
    return



if __name__ == '__main__':

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(1)

    opts = sys.argv[1]
    filenames = sys.argv[2:]

    write(opts, filenames)


# End of file
