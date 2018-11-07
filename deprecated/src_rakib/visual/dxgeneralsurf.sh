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
# Create OpenDX .general file for combined Citcom surf data
#

if [ -z $1 ]; then
    echo "usage: " `basename $0` file1 [file2 ...]
    exit
fi


for i; do

    if [ ! -f $i ]; then
	echo file \'$1\' not exist
	exit
    fi

    echo processing $i ...
    output=$i.general
    grid=`head -n 1 $i | awk '{print $3, $2, $1}'`

    echo file = $i > $output
    echo grid = $grid >> $output
    echo format = ascii >> $output
    echo interleaving = field >> $output
    echo majority = row >> $output
    echo header = lines 1 >> $output
    echo field = locations, topography, heatflux, surf_velocity >> $output
    echo structure = 2-vector, scalar, scalar, 2-vector >> $output
    echo type = float, float, float, float >> $output
    echo >> $output
    echo end >> $output

done


# version
# $Id: dxgeneralsurf.sh 2399 2005-10-04 23:58:43Z tan2 $

# End of file
