#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Copy modelname.log and modelname.time file to current directory
#
# Requirement: current working directory must be mounted on master_ip too.

if [ -z $2 ]; then
    echo "  usage:" `basename $0` modelname master_ip
    exit
fi

dir=`pwd`

rsh $2 cp $1.log $1.time $dir

# version
# $Id: getlog.sh,v 1.1 2004/01/14 23:22:52 tan2 Exp $

# End of file
