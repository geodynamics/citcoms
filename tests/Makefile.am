## Process this file with automake to produce Makefile.in
# $Id$

nodist_bin_SCRIPTS = \
	citcomsfull.sh \
	citcomsregional.sh \
	coupledcitcoms.sh
do_subst = sed \
	-e 's|[@]pkgpythondir[@]|$(pkgpythondir)|g' \
	-e 's|[@]PYTHON[@]|$(PYTHON)|g' \
	-e 's|[@]PYTHONPATH[@]|$(PYTHONPATH)|g'
citcomsfull.sh: $(srcdir)/citcomsfull.sh.in
	$(do_subst) < $(srcdir)/citcomsfull.sh.in > $@ || (rm -f $@ && exit 1)
citcomsregional.sh: $(srcdir)/citcomsregional.sh.in
	$(do_subst) < $(srcdir)/citcomsregional.sh.in > $@ || (rm -f $@ && exit 1)
coupledcitcoms.sh: $(srcdir)/coupledcitcoms.sh.in
	$(do_subst) < $(srcdir)/coupledcitcoms.sh.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_bin_SCRIPTS)
EXTRA_DIST = \
	citcomsfull.sh.in \
	citcomsregional.sh.in \
	coupledcitcoms.sh.in

# C++ tests
TESTS = $(EXTRA_PROGRAMS)
INSTALL_TESTS = $(EXTRA_PROGRAMS)
EXTRA_PROGRAMS = array2d
array2d_SOURCES = array2d.cc
PYTHON_BLDLIBRARY = -lpython$(PYTHON_VERSION)
INCLUDES = \
	-I$(top_srcdir)/lib/Common \
	$(MPIINCLUDES)
LDADD = \
	-lExchanger \
	-l_mpimodule -ljournal \
	$(PYTHON_BLDLIBRARY) \
	$(MPILIBS)
CC = $(MPICC)

EXTRA_DIST += \
	asap.py \
	exchange.py \
	hrothgar.py \
	signon.py \
	test1.sh \
	test2.sh \
	test5.sh

## end of Makefile.am
