## Process this file with automake to produce Makefile.in
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
	-e 's|[@]LAUNCHERS_PY[@]|$(pkgpythondir)/Components/Launchers.py|g' \
	-e 's|[@]MPIRUN[@]|$(MPIRUN)|g' \
	-e 's|[@]PYTHIA_MPIPYTHON[@]|$(PYTHIA_MPIPYTHON)|g' \
	-e 's|[@]QSUB[@]|$(QSUB)|g'
CitcomS.pml: $(srcdir)/CitcomS.pml.in
	$(do_subst) < $(srcdir)/CitcomS.pml.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_pkgsysconf_DATA)
EXTRA_DIST = CitcomS.pml.in

## end of Makefile.am
