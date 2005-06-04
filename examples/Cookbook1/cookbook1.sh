#!/bin/sh
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

../../tests/citcomsfull.sh \
\
--launcher.nodes=12 \
--launcher.nodegen="n%03d" \
--launcher.nodelist=[141-146,141-146] \
\
--steps=101 \
--controller.monitoringFrequency=25 \
\
--solver.datafile="/scratch/GURNIS/Test/cookbook1" \
\
--solver.ic.num_perturbations=1 \
--solver.ic.perturbl=3 \
--solver.ic.perturbm=2 \
\
--solver.rayleigh=5.0e+4
# End of file
