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
--steps=1000 \
\
--controller.monitoringFrequency=10 \
\
--launcher.nodes=64 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[141-172] \
\
--solver.datafile=./cookbook5_output/cookbook5 \
--solver.rayleigh=4.07e+08 \
\
--solver.bc.topvbc=1 \
--solver.param.file_vbcs=on \
--solver.param.vel_bound_file=./velocity/bvel.dat \
--solver.param.start_age=55 \
\
--solver.ic.restart=on \
--solver.ic.solution_cycles_init=0 \
--solver.datafile_old=./restart_files/cookbook5 \
\
--solver.mesher.coor=on \
--solver.mesher.coor_file=./coor.dat \
--solver.mesher.nprocx=2 \
--solver.mesher.nprocy=8 \
--solver.mesher.nprocz=4 \
--solver.mesher.nodex=17 \
--solver.mesher.nodey=65 \
--solver.mesher.nodez=33 \
--solver.mesher.theta_min=1.47 \
--solver.mesher.theta_max=1.67 \
--solver.mesher.fi_min=0 \
--solver.mesher.fi_max=0.5 \
--solver.mesher.radius_inner=0.7 \
\
--solver.const.refvisc=4e+21 \
--solver.visc.num_mat=4 \
--solver.visc.visc0=100,0.003,1,2 \
--solver.visc.TDEPV=on \
--solver.visc.viscE=24,24,24,24 \
--solver.visc.viscT=0.182,0.182,0.182,0.182 \
--solver.visc.viscZ=357.6,357.6,357.6,357.6 \
--solver.visc.VMIN=on \
--solver.visc.visc_min=0.01 \
--solver.visc.VMAX=on \
--solver.visc.visc_max=100.0 \
\
# version
# $Id: cookbook5.sh,v 1.2 2005/06/08 04:02:59 tan2 Exp $
\
# End of file
