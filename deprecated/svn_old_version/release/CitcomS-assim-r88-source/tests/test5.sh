#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# coupled citcoms test
# 2-proc coarse-grid solver, 2-proc fine-grid solver


FINE=fine
COARSE=coarse
OUTPUT=zzz

rm $FINE.* $COARSE.*


../bin//citcoms --coupled  \
--layout.coarse=[0-1] \
--layout.fine=[2-3] \
\
--coarse=regional \
--coarse.mesher=regional-sphere \
--coarse.datafile=$COARSE \
--coarse.mesher.nprocx=1 \
--coarse.mesher.nprocy=1 \
--coarse.mesher.nprocz=2 \
--coarse.mesher.nodex=5 \
--coarse.mesher.nodey=5 \
--coarse.mesher.nodez=5 \
--coarse.mesher.theta_max=1.7 \
--coarse.mesher.theta_min=1.4 \
--coarse.mesher.fi_max=1.3 \
--coarse.mesher.fi_min=1.0 \
--coarse.mesher.radius_inner=0.5 \
--coarse.mesher.radius_outer=1.0 \
\
--fine.mesher.nprocz=2 \
--fine.datafile=$FINE \
--fine.mesher.nodex=5 \
--fine.mesher.nodey=5 \
--fine.mesher.nodez=7 \
--fine.mesher.theta_max=1.6 \
--fine.mesher.theta_min=1.5 \
--fine.mesher.fi_max=1.2 \
--fine.mesher.fi_min=1.1 \
--fine.mesher.radius_inner=0.7 \
--fine.mesher.radius_outer=0.9 \
\
--fine.bc.side_sbcs=on \
\
--coarse.vsolver.accuracy=1e-3 \
--fine.vsolver.accuracy=1e-3 \
\
--fge.excludeTop=on \
--fge.excludeBottom=on \
\
--steps=1 \
--controller.monitoringFrequency=1 \
\
--journal.debug.Exchanger=on \
--journal.debug.CitcomS-Exchanger=on \
\
--journal.info.CitcomS-Interior-BBox=off \
--journal.info.CitcomS-Boundary-BBox=off \
--journal.info.CitcomS-Boundary-X=on \
--journal.info.CitcomS-Boundary-normal=on \
\
#| tee $OUTPUT

#> $OUTPUT



# version
# $Id: test5.sh 13196 2008-10-29 23:17:11Z tan2 $

# End of file
