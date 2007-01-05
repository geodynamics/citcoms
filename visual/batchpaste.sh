#!/bin/sh
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
# Run 'pasteCitcomData.sh' in a batch process
#
# Requirement: 1) current working directory must be mounted on master_ip too.
#              2) the list of ip has the same order as the MPI machinefile

if [ -z $4 ]; then
    echo "Usage:" `basename $0` datadir datafile timestep ip1 [ip2 ... ]
    exit
fi

paste_exe=`which execpaste.py`
cwd=`pwd`
datadir=$1
datafile=$2
timestep=$3
rank=0

while [ x"$4" != "x" ]
do
    cmd_paste="$paste_exe $datadir $datafile $rank $timestep $cwd"
    if [ x"$4" = x"$HOSTNAME" -o  $4 = "localhost" ]; then
	$cmd_paste
    else
        rsh $4 "$cmd_paste"
    fi

    shift
    rank=$(($rank+1))
done


# version
# $Id$

# End of file
