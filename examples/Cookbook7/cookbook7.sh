#!/bin/sh

./citcomsregional.sh \
--staging.nodegen=n%03d \
--staging.nodelist=[135-140,165-173,176,135-140,165-173,176] \
--staging.nodes=32 \
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

