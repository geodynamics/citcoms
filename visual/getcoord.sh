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
proc=0

while [ "$3" ]
do
    cmd_copy="cp $modelname.coord.$proc $cwd"
    rsh $3 "cd $modeldir; $cmd_copy"
    shift
    let proc=proc+1
done


# version
# $Id: getcoord.sh,v 1.1 2004/06/08 01:32:39 tan2 Exp $

# End of file
