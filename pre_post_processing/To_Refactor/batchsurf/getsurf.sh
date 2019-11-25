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
# Copy the surf files from remote machines to the current directory
#
# Requirement: 1) current working directory must be mounted on ip* too.
#              2) the list of ip has the same order as the MPI machinefile

if [ -z $5 ]; then
    echo "Usage:" `basename $0` modeldir modelname timestep nprocz ip1 [ip2 ... ]
    exit
fi

cwd=`pwd`
modeldir=$1
modelname=$2
timestep=$3
nprocz=$4
let proc=nprocz-1

while [ "$5" ]
do
    cmd_copy="cp $modelname.surf.$proc.$timestep $cwd"

    if [ $5 == $HOSTNAME -o $5 == "localhost" ]; then
        # using subshell, so that our working directory is unchanged
	(cd $modeldir; $cmd_copy)
    else
	rsh $5 "cd $modeldir; $cmd_copy"
    fi

    shift
    let proc=proc+nprocz
done


# version
# $Id: getsurf.sh 4592 2006-09-22 22:14:37Z tan2 $

# End of file
