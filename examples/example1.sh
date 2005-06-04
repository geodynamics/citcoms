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
--launcher.nodes=4 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[131-132] \
\
--solver.datafile=example1 \
\
--solver.mesher.nprocx=2 \
--solver.mesher.nprocy=2 \
--solver.mesher.nodex=17 \
--solver.mesher.nodey=17 \
--solver.mesher.nodez=9 \


# version
# $Id: example1.sh,v 1.5 2005/06/03 21:51:40 leif Exp $

# End of file
