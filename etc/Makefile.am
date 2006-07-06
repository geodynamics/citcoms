## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
## Copyright (C) 2002-2005, California Institute of Technology.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##</LicenseText>

# $Id$

dist_pkgsysconf_DATA = \
	full.odb \
	full-sphere.odb \
	globus.odb \
	incomp-newtonian.odb \
	incomp-non-newtonian.odb \
	lam-mpi.odb \
	lsf.odb \
	mpich.odb \
	pbs.odb \
	regional.odb \
	regional-sphere.odb \
	temp.odb \
	__vault__.odb

nodist_pkgsysconf_DATA = CitcomS.pml
do_subst = sed \
	-e 's|[@]BSUB[@]|$(BSUB)|g' \
	-e 's|[@]GLOBUSRUN[@]|$(GLOBUSRUN)|g' \
	-e 's|[@]INPUT[@]|$(pkgsysconfdir)/CitcomS.pml: generated from CitcomS.pml.in|g' \
	-e 's|[@]LAUNCHER[@]|$(LAUNCHER)|g' \
	-e 's|[@]LAUNCHERS_PY[@]|$(pkgpyexecdir)/Components/Launchers.py|g' \
	-e 's|[@]MPIRUN[@]|$(MPIRUN)|g' \
	-e 's|[@]QSUB[@]|$(QSUB)|g'
CitcomS.pml: $(srcdir)/CitcomS.pml.in
	$(do_subst) < $(srcdir)/CitcomS.pml.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_pkgsysconf_DATA)
EXTRA_DIST = CitcomS.pml.in

## end of Makefile.am
