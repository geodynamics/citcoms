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
--steps=1 \
\
--controller.monitoringFrequency=10 \
\
--staging.nodes=4 \
--staging.nodegen="n%03d" \
--staging.nodelist=[131-132] \
\
--solver.datafile=example1 \
\
--solver.mesher.nprocx=2 \
--solver.mesher.nprocy=2 \
--solver.mesher.nodex=17 \
--solver.mesher.nodey=17 \
--solver.mesher.nodez=9 \


# version
# $Id: example1.sh,v 1.4 2005/05/27 21:41:22 vlad Exp $

# End of file
