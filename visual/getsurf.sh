#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
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
    rsh $5 "cd $modeldir; $cmd_copy"
    shift
    let proc=proc+nprocz
done


# version
# $Id: getsurf.sh,v 1.1 2004/06/08 01:35:06 tan2 Exp $

# End of file
