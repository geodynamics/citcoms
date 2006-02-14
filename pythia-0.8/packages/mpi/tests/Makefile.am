## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/09 16:12:05 leif Exp $

TESTS =
INSTALL_TESTS = \
	basic.py \
	cartesian.py \
	groups.py \
	ports.py \
	raw.py \
	signon.py \
	simple.py
TESTS_ENVIRONMENT = mpirun -np 4 $(bindir)/mpipython.exe
EXTRA_DIST = $(INSTALL_TESTS)

installcheck-local:
	$(MAKE) $(AM_MAKEFLAGS) TESTS="$(INSTALL_TESTS)" check

## end of Makefile.am
