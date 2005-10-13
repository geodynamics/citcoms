## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/20 23:33:50 leif Exp $

installcheck-local:
	$(MAKE) $(AM_MAKEFLAGS) TESTS="$(INSTALL_TESTS)" check

# C++ tests
EXTRA_PROGRAMS = \
	canonicalMesh \
	tuple \
	vector
TESTS = $(EXTRA_PROGRAMS)
INSTALL_TESTS = $(EXTRA_PROGRAMS)
XFAIL_TESTS =
EXTRA_DIST = \
	libpyre/manip1.cc \
	libpyre/manip2.cc \
	libpyre/manip3.cc \
	libpyre/point.cc \
	libpyre/recycler.cc \
	libpyre/shareable.cc
INCLUDES = -I$(top_builddir)/include -I$(PYTHON_INCDIR)
LDADD = $(top_builddir)/packages/journal/libjournal/libjournal.la
canonicalMesh_SOURCES = libpyre/canonicalMesh.cc
tuple_SOURCES = libpyre/tuple.cc
vector_SOURCES = libpyre/vector.cc

# Python tests
PYTHON_TESTS = \
	applications/daemon.py \
	applications/hello.py \
	applications/load.py \
	geometry/mesh.py \
	geometry/pickle.py \
	geometry/unpickle.py \
	idd/idd.py \
	inventory/array.py \
	inventory/bool.py \
	inventory/component.py \
	inventory/curator.py \
	inventory/inventory.py \
	inventory/logicals.py \
	inventory/odb.py \
	inventory/outfile.py \
	inventory/trait.py \
	inventory/vectors.py \
	ipa/ipa.py \
	ipc/connection.py \
	ipc/monitor.py \
	odb/contacts/raw.py \
	services/harness.py \
	simulations/simulation.py \
	util/expand.py \
	weaver/languages.py \
	weaver/mills.py
INSTALL_TESTS += $(PYTHON_TESTS)
XFAIL_TESTS += \
	inventory/inventory.py \
	inventory/outfile.py \
	inventory/trait.py \
	ipc/monitor.py \
	odb/contacts/raw.py \
	services/harness.py
EXTRA_DIST += \
	$(PYTHON_TESTS) \
	applications/harness.py \
	db/create.py \
	db/db.py \
	inventory/infile.py \
	ipc/selector.py \
	odb/contacts/Person.py \
	services/echo-client.py

# input files
INPUT_FILES = \
	applications/hello.pml \
	applications/morning.odb \
	geometry/canister.odb \
	ipa/ipa.pml \
	ipa/userdb.crypt \
	ipa/userdb.md5 \
	ipa/userdb.sha \
	odb/contacts/10000.odb \
	odb/contacts/10001.odb \
	services/echo.odb \
	services/idd.odb \
	services/ipa.odb \
	services/journal.odb \
	services/userdb.md5
EXTRA_DIST += $(INPUT_FILES)
BUILT_SOURCES = \
	hello.pml \
	morning.odb \
	canister.odb \
	ipa.pml \
	userdb.crypt \
	userdb.md5 \
	userdb.sha \
	10000.odb \
	10001.odb \
	echo.odb \
	idd.odb \
	ipa.odb \
	journal.odb \
	userdb.md5
CLEANFILES = $(BUILT_SOURCES)
hello.pml morning.odb:
	cp -f $(srcdir)/applications/$@ $@
canister.odb:
	cp -f $(srcdir)/geometry/$@ $@
ipa.pml userdb.crypt userdb.md5 userdb.sha:
	cp -f $(srcdir)/ipa/$@ $@
10000.odb 10001.odb:
	cp -f $(srcdir)/odb/contacts/$@ $@
echo.odb idd.odb ipa.odb journal.odb userdb.md5:
	cp -f $(srcdir)/services/$@ $@

## end of Makefile.am
