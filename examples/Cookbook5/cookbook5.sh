#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

citcomsregional.sh \
\
--steps=1000 \
\
--controller.monitoringFrequency=10 \
\
--launcher.nodes=64 \
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
--solver.visc.visc_max=100.0

# version
# $Id$

# End of file
