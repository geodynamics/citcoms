#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Create OpenDX .general file for combined Citcom Data
#

if [ -z $1 ]; then
    echo "  usage:" `basename $0` file1 [file2 ...]
    exit
fi


for i; do

    if [ ! -f $i ]; then
	echo file \'$1\' not exist
	exit
    fi

    echo processing $i ...
    output=$i.general
    grid=`head -1 $i`

    echo file = $i > $output
    echo grid = $grid >> $output
    echo format = ascii >> $output
    echo interleaving = field >> $output
    echo majority = row >> $output
    echo header = lines 1 >> $output
    echo field = locations, velocity, temperature, viscosity >> $output
    echo structure = 3-vector, 3-vector, scalar, scalar >> $output
    echo type = float, float, float, float >> $output
    echo >> $output
    echo end >> $output

done


# version
# $Id: dxgeneral.sh,v 1.2 2004/01/15 00:17:39 tan2 Exp $

# End of file
