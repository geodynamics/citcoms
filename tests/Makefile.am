## Process this file with automake to produce Makefile.in
# $Id$

nodist_bin_SCRIPTS = \
	citcomsfull.sh \
	citcomsregional.sh
do_subst = sed \
	-e 's|[@]pkgpythondir[@]|$(pkgpythondir)|g' \
	-e 's|[@]PYTHON[@]|$(PYTHON)|g' \
	-e 's|[@]PYTHONPATH[@]|$(PYTHONPATH)|g'
citcomsfull.sh: $(srcdir)/citcomsfull.sh.in
	$(do_subst) < $(srcdir)/citcomsfull.sh.in > $@ || (rm -f $@ && exit 1)
citcomsregional.sh: $(srcdir)/citcomsregional.sh.in
	$(do_subst) < $(srcdir)/citcomsregional.sh.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_bin_SCRIPTS)
EXTRA_DIST = \
	citcomsfull.sh.in \
	citcomsregional.sh.in

EXTRA_DIST += \
	asap.py \
	hrothgar.py \
	signon.py

## end of Makefile.am
