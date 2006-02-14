## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/09 16:12:06 leif Exp $

TESTS =
INSTALL_TESTS = \
	cmdline.py \
	hello.py \
	html.py \
	login.py \
	simple.py
EXTRA_DIST = $(INSTALL_TESTS)

installcheck-local:
	$(MAKE) $(AM_MAKEFLAGS) TESTS="$(INSTALL_TESTS)" check

## end of Makefile.am
