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
    echo "  usage:" `basename $0` filename
    exit
fi

if [ ! -f $1 ]; then
    echo file \'$1\' not exist
    exit
fi

output=$1.general
grid=`head -1 $1`

echo file = $1 > $output
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


# version
# $Id: dxgeneral.sh,v 1.1 2004/01/14 23:22:52 tan2 Exp $

# End of file
