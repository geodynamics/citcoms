#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# $Id$

# End of file
