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

# functional test of exchanger
# 12-proc coarse-grid exchanger, 1-proc fine-grid exchanger


OUTPUT=zzz
TEMPFILE=/tmp/$USER/tmptest1


exchange.py \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[101-113,115-129] \
--launcher.nodes=13 \
--layout.coarse=[0-11] \
--layout.fine=[12] \
> $OUTPUT


echo 'coarse exchanger: rank=8  leader=11  remoteLeader=12' > $TEMPFILE
result=Failed
if grep 'coarse exchanger: rank=8' $OUTPUT | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test1: coarse exchanger rank ... $result.


echo 'fine exchanger: rank=12  leader=0  remoteLeader=11' > $TEMPFILE
result=Failed
if grep 'fine exchanger: rank=12' $OUTPUT | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test1: fine exchanger rank ... $result.


echo ' -- in Boundary::Boundary  size = 44' > $TEMPFILE
result=Failed
if grep 'in Boundary::Boundary  size = 44' $OUTPUT | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test1: boundary size ... $result.


echo " --   X:  31:  1.7 2.1 0.9" > $TEMPFILE
result=Failed
if grep 'X:  31:' $OUTPUT | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test1: boundary coord ... $result.


echo "0" > $TEMPFILE
result=Failed
if grep 'proc:' $OUTPUT | grep 12$ | wc -l | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test1: bid2proc ... $result.


rm $TEMPFILE


# version
# $Id: test1.sh,v 1.4 2005/06/10 02:23:24 leif Exp $

# End of file
