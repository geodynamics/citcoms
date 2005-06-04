#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

../../tests/citcomsregional.sh \
\
--steps=61 \
\
--controller.monitoringFrequency=30 \
\
--launcher.nodes=4 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[101-102,101-102] \
\
--solver.datafile=/scratch/username/cookbook2 \
\
--solver.mesher.nprocx=2 \
--solver.mesher.nprocy=2 \
--solver.mesher.nodex=17 \
--solver.mesher.nodey=17 \
--solver.mesher.nodez=9 \
\
--solver.bc.topvbc=1 \
--solver.bc.topvbxval=100 \
--solver.bc.topvbyvel=0 \
\
--solver.ic.num_perturbations=1 \
--solver.ic.perturbmag=0.0

# version
# $Id: cookbook2.sh,v 1.2 2005/06/03 21:51:40 leif Exp $

# End of file
