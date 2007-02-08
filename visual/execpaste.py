#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan
# Copyright (C) 2006, California Institute of Technology.
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

"""Execute pasteCitcomData.sh to retrieve the pasted data

Usage: execpaste.py datadir datafile processor_rank timestep save_dir

datadir: same input parameter for CitcomS
datafile: same input parameter for CitcomS
processor_rank: MPI rank of current processor
timestep: timestep to retrieve
save_dir: which directory to save the retrieved data
"""

import sys

if len(sys.argv) != 6:
    print __doc__
    sys.exit()

datadir = sys.argv[1]
datafile = sys.argv[2]
rank = sys.argv[3]
timestep = sys.argv[4]
save_dir = sys.argv[5]

import os
paste_exe = "pasteCitcomData.sh"


## expand datadir
s = "%HOSTNAME"
try:
    datadir.index(s)
except: pass
else:
    from socket import gethostname
    datadir = datadir.replace(s, gethostname())

s = "%RANK"
try:
    datadir.index(s)
except: pass
else:
    datadir = datadir.replace(s, rank)

if datadir == "%DATADIR":
    fp = os.popen("citcoms_datadir", "r")
    datadir = fp.readline().strip()
    fp.close()


## run paste_exe and copy the pasted data to save_dir
os.chdir(datadir)
cmd = """
%(paste_exe)s %(datafile)s %(rank)s %(timestep)s && \
cp %(datafile)s.%(rank)s.%(timestep)s %(save_dir)s
""" % vars()
os.system(cmd)


# version
# $Id$

# End of file
