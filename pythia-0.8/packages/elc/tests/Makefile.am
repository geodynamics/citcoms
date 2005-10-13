## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/09 16:12:04 leif Exp $

TESTS =
INSTALL_TESTS = \
	exchange.py \
	signon.py
EXTRA_DIST = \
	$(INSTALL_TESTS) \
	hollow-cube.odb

installcheck-local:
	$(MAKE) $(AM_MAKEFLAGS) TESTS="$(INSTALL_TESTS)" check

## end of Makefile.am
