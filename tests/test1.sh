#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# functional test of exchanger
# 12-proc coarse-grid exchanger, 1-proc fine-grid exchanger


OUTPUT=zzz
TEMPFILE=/tmp/$USER/tmptest1


exchange.py \
--staging.nodegen="n%03d" \
--staging.nodelist=[101-113,115-129] \
--staging.nodes=13 \
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
# $Id: test1.sh,v 1.2 2003/10/24 05:23:36 tan2 Exp $

# End of file
