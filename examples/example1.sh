#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

../tests/citcomsregional.sh \
\
--steps=71 \
\
--controller.monitoringFrequency=10 \
\
--staging.nodes=4 \
--staging.nodegen="n%03d" \
--staging.nodelist=[101-102] \
\
--solver.datafile=/scratch/MARIAS/Test/test1/test1 \
\
--solver.mesher.nprocx=2 \
--solver.mesher.nprocy=2 \
--solver.mesher.nodex=17 \
--solver.mesher.nodey=17 \
--solver.mesher.nodez=9 \


# version
# $Id: example1.sh,v 1.2 2004/06/28 18:10:56 tan2 Exp $

# End of file
