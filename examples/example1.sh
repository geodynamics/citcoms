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
--solver.mesher.nnodex=17 \
--solver.mesher.nnodey=17 \
--solver.mesher.nnodez=9 \


# version
# $Id: example1.sh,v 1.1 2004/06/24 19:38:57 tan2 Exp $

# End of file
