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
--steps=250 \
\
--controller.monitoringFrequency=10 \
\
--staging.nodes=16 \
--staging.nodegen="n%03d" \
--staging.nodelist=[150-157,150-157] \
\
--solver.rayleigh=1e6 \
--solver.datafile=../Cookbook4/cookbook4_output/cookbook4 \
\
--solver.ic.num_perturbations=1 \
--solver.ic.perturbmag=0.05 \
--solver.ic.perturbl=1 \
--solver.ic.perturbm=0 \
--solver.ic.perturblayer=10 \
\
--solver.mesher.coor=on \
--solver.mesher.coor_file=../Cookbook4/coor.dat \
--solver.mesher.nprocx=4 \
--solver.mesher.nprocy=2 \
--solver.mesher.nprocz=2 \
--solver.mesher.nodex=33 \
--solver.mesher.nodey=17 \
--solver.mesher.nodez=17 \
--solver.mesher.levels=1 \
--solver.mesher.theta_min=1 \
--solver.mesher.theta_max=2 \
--solver.mesher.fi_min=0 \
--solver.mesher.fi_max=1 \
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
\
# version
# $Id: cookbook4.sh,v 1.1 2005/01/18 22:30:29 vlad Exp $

# End of file
