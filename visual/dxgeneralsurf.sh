#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
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
    grid=`head -1 $i | awk '{print $3, $2, $1}'`

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
# $Id: dxgeneralsurf.sh,v 1.1 2004/06/08 01:35:06 tan2 Exp $

# End of file
