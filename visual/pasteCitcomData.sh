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
# Gather Citcom data (coordinate, velocity, temperature, viscosity) of current
# processor into a single file.

if [ -z $3 ]; then
    echo "  usage:" `basename $0` datafile processor_rank timestep
    exit
fi

datafile=$1
rank=$2
step=$3

line=`cat $datafile.velo.$rank.$step | wc -l`
let line=line-1
#echo $line

tail -n $line $datafile.velo.$rank.$step \
  | paste -d' ' $datafile.coord.$rank - $datafile.visc.$rank.$step \
    > $datafile.$rank.$step


# version
# $Id$

# End of file
