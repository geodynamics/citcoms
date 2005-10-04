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
--steps=250 \
\
--controller.monitoringFrequency=10 \
\
--launcher.nodes=16 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[150-157,150-157] \
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
# $Id$

# End of file
