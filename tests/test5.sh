#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
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


./coupledcitcoms.sh  \
--launcher.nodes=4 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[171-172,171-172] \
\
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
--coarse.vsolver.tole_compressibility=1e-3 \
--fine.vsolver.accuracy=1e-3 \
--fine.vsolver.tole_compressibility=1e-3 \
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
# $Id: test5.sh,v 1.3 2005/06/10 02:23:24 leif Exp $

# End of file
