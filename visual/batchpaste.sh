#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
#
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
#
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Run 'pasteCitcomData.sh' in a batch process
#
# Requirement: 1) current working directory must be mounted on master_ip too.
#              2) the list of ip has the same order as the MPI machinefile

if [ -z $4 ]; then
    echo "Usage:" `basename $0` modeldir modelname timestep ip1 [ip2 ... ]
    exit
fi

paste_exe=`which pasteCitcomData.sh`
cwd=`pwd`
modeldir=$1
modelname=$2
timestep=$3
n=0

while [ "$4" ]
do
    cmd_paste="$paste_exe $modelname $n $timestep"
    cmd_copy="cp $modelname.$n.$timestep $cwd"

    if [ $4 == $HOSTNAME -o $4 == "localhost" ]; then
	cd $modeldir && $cmd_paste && $cmd_copy
    else
 	rsh $4 "cd $modeldir && $cmd_paste && $cmd_copy"
    fi

    shift
    let n=n+1
done


# version
# $Id: batchpaste.sh,v 1.6 2005/07/19 20:55:47 leif Exp $

# End of file
