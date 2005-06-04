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
--steps=200 \
\
--controller.monitoringFrequency=25 \
\
--launcher.nodes=4 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[101-102,101-102] \
\
--solver.rayleigh=1e6 \
--solver.datafile=/scratch/username/cookbook3 \
\
--solver.mesher.nprocx=2 \
--solver.mesher.nprocy=2 \
--solver.mesher.nodex=17 \
--solver.mesher.nodey=17 \
--solver.mesher.nodez=9 \
\
--solver.visc.VISC_UPDATE=on \
--solver.visc.num_mat=4 \
--solver.visc.visc0=1,1,1,1 \
--solver.visc.TDEPV=on \
--solver.visc.viscE=0.2,0.2,0.2,0.2 \
--solver.visc.viscT=0,0,0,0 \
--solver.visc.VMIN=on \
--solver.visc.visc_min=1.0 \
--solver.visc.VMAX=on \
--solver.visc.visc_max=100.0

# version
# $Id: cookbook3.sh,v 1.2 2005/06/03 21:51:41 leif Exp $

# End of file
