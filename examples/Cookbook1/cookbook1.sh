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
--staging.nodes=12 \
--staging.nodegen="n%03d" \
--staging.nodelist=[101-106,101-106] \
\
--steps=101 \
--controller.monitoringFrequency=25 \
\
--solver.datafile="/scratch/username/cookbook1" \
\
--solver.ic.num_perturbations=1 \
--solver.ic.perturbl=8 \
--solver.ic.perturbm=8 

# version
# $Id: cookbook1.sh,v 1.1 2004/06/29 17:25:12 tan2 Exp $

# End of file
