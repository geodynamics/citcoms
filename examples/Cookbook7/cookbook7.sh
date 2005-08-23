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

./citcomsregional.sh \
--launcher.nodegen=n%03d \
--launcher.nodelist=[135-140,165-173,176,135-140,165-173,176] \
--launcher.nodes=32 \
\
--solver.datafile=/scratch/username/cookbook7 \
--solver.rayleigh=4.312616e+08 \
\
--solver.mesher.nprocx=4 \
--solver.mesher.nprocy=4 \
--solver.mesher.nprocz=2 \
--solver.mesher.nodex=61 \
--solver.mesher.nodey=61 \
--solver.mesher.nodez=25 \
--solver.mesher.coor=true \
--solver.mesher.coor_file="./coord.dat" \
\
--solver.tsolver.fixed_timestep=7.77e-10 \
\
--solver.vsolver.precond=on \
--solver.vsolver.accuracy=1e-10 \
--solver.vsolver.tole_compressibility=1e-06 \
--solver.vsolver.mg_cycle=1 \
--solver.vsolver.down_heavy=5 \
--solver.vsolver.up_heavy=5 \
--solver.vsolver.vlowstep=100000 \
--solver.vsolver.vhighstep=3 \
--solver.vsolver.piterations=100000 \
\
--solver.ic.tic_method=2 \
--solver.ic.num_perturbations=0 \
--solver.ic.half_space_age=100.0 \
--solver.ic.blob_center=1.570800e+00,1.570800e+00,9.246600e-01 \
--solver.ic.blob_radius=6.278334e-02 \
--solver.ic.blob_dT=0.18 \
\
--solver.bc.topvbc=2 \
--solver.bc.pseudo_free_surf=on \
--solver.bc.bottbcval=0.82 \
\
--solver.param.start_age=60 \
--solver.param.mantle_temp=0.82 \
\
--solver.visc.TDEPV=on \
--solver.visc.visc0=1,1,1,1 \
--solver.visc.viscE=9.50614,9.50614,9.50614,9.50614 \
--solver.visc.viscT=1.02126,1.02126,1.02126,1.02126 \
\
--solver.const.layerd=6.371e+06 \
--solver.const.density=3270.0 \
--solver.const.thermdiff=1.0e-06 \
--solver.const.gravacc=10.0 \
--solver.const.thermexp=3.0e-05 \
--solver.const.refvisc=1.0e+21 \
\
--steps=100 \
--controller.monitoringFrequency=10 \

