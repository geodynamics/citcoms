#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Gather Citcom data (coordinate, velocity, temperature, viscosity) of current
# processor into a single file.

if [ -z $3 ]; then
    echo "  usage:" `basename $0` modelname processor_no time_step
    exit
fi

line=`cat $1.velo.$2.$3 | wc -l`
let line=line-1
#echo $line

tail -$line $1.velo.$2.$3 | paste -d' ' $1.coord.$2 - $1.visc.$2.$3 > $1.$2.$3


# version
# $Id: pasteCitcomData.sh,v 1.2 2004/02/05 00:46:37 tan2 Exp $

# End of file
