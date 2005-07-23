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
citcomsregional.sh \
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
# $Id: cookbook5.sh,v 1.3.2.1 2005/07/23 02:02:49 leif Exp $
\
# End of file
