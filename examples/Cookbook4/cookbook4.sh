#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

../../tests/citcomsregional.sh \
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
# $Id: cookbook4.sh,v 1.3 2005/06/10 02:23:13 leif Exp $

# End of file
