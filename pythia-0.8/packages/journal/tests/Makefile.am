## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:49 leif Exp $

installcheck-local:
	$(MAKE) $(AM_MAKEFLAGS) TESTS="$(INSTALL_TESTS)" check

# C++ tests
EXTRA_PROGRAMS = \
	control \
	debug \
	diagnostics \
	error \
	firewall \
	infox \
	null \
	state \
	warning
TESTS = $(EXTRA_PROGRAMS)
INSTALL_TESTS = $(EXTRA_PROGRAMS)
XFAIL_TESTS =
INCLUDES = -I$(top_builddir)/include -I$(PYTHON_INCDIR)
LDADD = ../libjournal/libjournal.la
control_SOURCES = libjournal/control.cc
debug_SOURCES = libjournal/debug.cc
diagnostics_SOURCES = libjournal/diagnostics.cc
error_SOURCES = libjournal/error.cc
firewall_SOURCES = libjournal/firewall.cc
infox_SOURCES = libjournal/info.cc
null_SOURCES = libjournal/null.cc
state_SOURCES = libjournal/state.cc
warning_SOURCES = libjournal/warning.cc

# Python tests
PYTHON_TESTS = \
	components/journald.py \
	components/remote.py \
	journal/channels.py \
	journal/diagnostics.py \
	journal/info.py \
	journal/renderer.py \
	journal/signon.py
INSTALL_TESTS += $(PYTHON_TESTS)
XFAIL_TESTS += journal/diagnostics.py
EXTRA_DIST = $(PYTHON_TESTS)

## end of Makefile.am
