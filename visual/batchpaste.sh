#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Run 'pasteCitcomData.sh' in a batch process
#
# Requirement: 1) current working directory must be mounted on master_ip too.
#              2) the list of ip has the same order as the MPI machinefile

if [ -z $4 ]; then
    echo "  usage:" `basename $0` modeldir modelname timestep ip1 [ip2 ... ]
    exit
fi

# hard-wired path, to be fixed
mydir=$EXPORT_ROOT/packages/CitcomS/visual

cwd=`pwd`
modeldir=$1
modelname=$2
timestep=$3
n=0

while [ "$4" ]
do
    cmd_paste="$mydir/pasteCitcomData.sh $modelname $n $timestep"
    cmd_copy="cp $modelname.$n.$timestep $cwd"
    rsh $4 "cd $modeldir; $cmd_paste; $cmd_copy" &
    shift
    let n=n+1
done


# version
# $Id: batchpaste.sh,v 1.1 2004/01/14 23:22:52 tan2 Exp $

# End of file
