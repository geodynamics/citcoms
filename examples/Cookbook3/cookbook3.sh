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
# $Id$

# End of file
