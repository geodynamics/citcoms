#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Copy the coord files from remote machines to the current directory
#
# Requirement: 1) current working directory must be mounted on ip* too.
#              2) the list of ip has the same order as the MPI machinefile

if [ -z $3 ]; then
    echo "Usage:" `basename $0` modeldir modelname ip1 [ip2 ... ]
    exit
fi

cwd=`pwd`
modeldir=$1
modelname=$2
nprocz=$3
let proc=nprocz-1

while [ "$4" ]
do
    cmd_copy="cp $modelname.coord.$proc $cwd"
    rsh $4 "cd $modeldir; $cmd_copy"
    shift
    let proc=proc+nprocz
done


# version
# $Id: getcoord.sh,v 1.2 2004/09/21 23:44:12 ces74 Exp $

# End of file
