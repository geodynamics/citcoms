#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# coupled citcoms test
# 4-proc coarse-grid solver, 1-proc fine-grid solver
# no correction on outflow


FINE=fine
COARSE=coarse
OUTPUT=zzz
TEMPFILE=/tmp/$USER/tmptest2

rm $FINE.* $COARSE.*


coupledcitcoms.py  \
--staging.nodegen="n%03d" \
--staging.nodelist=[115-129,131-170] \
--staging.nodes=5 \
--layout.coarse=[0-3] \
--layout.fine=[4] \
--coarse=regional \
--coarse.mesher=regional-sphere \
--coarse.datafile=$COARSE \
--coarse.mesher.nprocx=2 \
--coarse.mesher.nprocy=2 \
--coarse.mesher.nprocz=1 \
--coarse.mesher.nodex=9 \
--coarse.mesher.nodey=9 \
--coarse.mesher.nodez=9 \
--coarse.mesher.theta_max=1.795 \
--coarse.mesher.theta_min=1.345 \
--coarse.mesher.fi_max=1.795 \
--coarse.mesher.fi_min=1.345 \
--coarse.mesher.radius_inner=0.55 \
--coarse.mesher.radius_outer=1.0 \
--fine.datafile=$FINE \
--fine.mesher.nodex=17 \
--fine.mesher.nodey=17 \
--fine.mesher.nodez=17 \
--fine.mesher.theta_max=1.616875 \
--fine.mesher.theta_min=1.466875 \
--fine.mesher.fi_max=1.616875 \
--fine.mesher.fi_min=1.466875 \
--fine.mesher.radius_inner=0.709375 \
--fine.mesher.radius_outer=0.859375 \
--fine.bc.toptbc=0 \
--fine.bc.toptbcval=0 \
--fine.bc.bottbc=0 \
--fine.bc.bottbcval=0 \
--fine.vsolver.precond=False \
--fine.vsolver.accuracy=1e-9 \
--fine.vsolver.tole_compressibility=1e-6 \
--fine.vsolver.piterations=8000 \
--fine.vsolver.vlowstep=5000 \
--fine.vsolver.vhighstep=10 \
--steps=1 \
--controller.monitoringFrequency=1 \
> $OUTPUT


echo '1.61688 1.51375 0.765625' > $TEMPFILE
result=Failed
if grep 'X:' $OUTPUT | grep ' 183:' | awk '{print $3 " " $4 " " $5}' \
    | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: boundary.X ... $result.


echo '2755' > $TEMPFILE
result=Failed
if grep 'bid:' $OUTPUT | grep ' 1344:' | awk '{print $3}' \
    | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: bid2gid ... $result.


echo '1' > $TEMPFILE
if grep 'proc:' $OUTPUT | grep ' 954:' | awk '{print $3}' \
    | diff -w - $TEMPFILE; then

    echo '3' > $TEMPFILE
    if grep 'proc:' $OUTPUT | grep ' 955:' | awk '{print $3}' \
	| diff -w - $TEMPFILE; then
    result=Passed
    fi
fi
echo test2: bid2proc ... $result.


echo '1.87341' > $TEMPFILE
result=Failed
if grep 'before boundary' $OUTPUT | cut -c 49- | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: outflow1 ... $result.


echo '-1.12559e-15' > $TEMPFILE
result=Failed
if grep 'after boundary' $OUTPUT | cut -c 67- | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: outflow2 ... $result.


echo '9.787985e+02 9.785439e+02 1.654943e+03 0.000000e+00' > $TEMPFILE
result=Failed
if tail +3 $FINE.velo.0.0 | head -1 | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: imposeBC ... $result.


echo '<stdin>: N = 4913       <-2265.95/1950.77>      <-2264.66/1949> <944.907/17056.7>       <0/1>' > $TEMPFILE
result=Failed
if tail +3 $FINE.velo.0.0 | minmax | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: minmax ... $result.


echo '-6.325670e+02 -3.417676e+02 1.111416e+04 4.940558e-01' > $TEMPFILE
result=Failed
if tail +2120 $FINE.velo.0.0 | head -1 | diff -w - $TEMPFILE; then
    result=Passed
fi
echo test2: content ... $result.


rm $TEMPFILE


# version
# $Id: test2.sh,v 1.1 2003/10/21 17:31:10 tan2 Exp $

# End of file
