#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# coupled citcoms test
# 2-proc coarse-grid solver, 2-proc fine-grid solver


FINE=fine
COARSE=coarse
OUTPUT=zzz

rm $FINE.* $COARSE.*


./coupledcitcoms.sh  \
--staging.nodes=4 \
--staging.nodegen="n%03d" \
--staging.nodelist=[171-172,171-172] \
\
--layout.coarse=[0-1] \
--layout.fine=[2-3] \
\
--coarse=regional \
--coarse.mesher=regional-sphere \
--coarse.datafile=$COARSE \
--coarse.mesher.nprocx=1 \
--coarse.mesher.nprocy=1 \
--coarse.mesher.nprocz=2 \
--coarse.mesher.nodex=5 \
--coarse.mesher.nodey=5 \
--coarse.mesher.nodez=5 \
--coarse.mesher.theta_max=1.7 \
--coarse.mesher.theta_min=1.4 \
--coarse.mesher.fi_max=1.3 \
--coarse.mesher.fi_min=1.0 \
--coarse.mesher.radius_inner=0.5 \
--coarse.mesher.radius_outer=1.0 \
\
--fine.mesher.nprocz=2 \
--fine.datafile=$FINE \
--fine.mesher.nodex=5 \
--fine.mesher.nodey=5 \
--fine.mesher.nodez=7 \
--fine.mesher.theta_max=1.6 \
--fine.mesher.theta_min=1.5 \
--fine.mesher.fi_max=1.2 \
--fine.mesher.fi_min=1.1 \
--fine.mesher.radius_inner=0.7 \
--fine.mesher.radius_outer=0.9 \
\
--fine.bc.side_sbcs=on \
\
--coarse.vsolver.accuracy=1e-3 \
--coarse.vsolver.tole_compressibility=1e-3 \
--fine.vsolver.accuracy=1e-3 \
--fine.vsolver.tole_compressibility=1e-3 \
\
--fge.excludeTop=on \
--fge.excludeBottom=on \
\
--steps=1 \
--controller.monitoringFrequency=1 \
\
--journal.debug.Exchanger=on \
--journal.debug.CitcomS-Exchanger=on \
\
--journal.info.CitcomS-Interior-BBox=off \
--journal.info.CitcomS-Boundary-BBox=off \
--journal.info.CitcomS-Boundary-X=on \
--journal.info.CitcomS-Boundary-normal=on \
\
#| tee $OUTPUT

#> $OUTPUT



# version
# $Id: test5.sh,v 1.1 2005/05/18 01:57:44 tan2 Exp $

# End of file
