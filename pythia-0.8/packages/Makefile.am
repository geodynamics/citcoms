## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.2 2005/09/28 21:10:01 leif Exp $

if COND_MPI
    MAYBE_MPI = mpi elc pulse rigid
endif
if COND_ACIS
    MAYBE_ACIS = acis
endif
SUBDIRS = journal pyre blade merlin opal $(MAYBE_MPI) $(MAYBE_ACIS)

# Use a .pth file to add Pythia's packages to Python's sys.path
# without polluting the top-level 'site-packages' directory.
# Following the example of PyGTK, the Pythia subdirectory includes a
# version number, potentially allowing multiple versions of Pythia to
# co-exist.  Again following the example of PyGTK, the version is
# intentionally omitted from the .pth filename; only the most recently
# installed version of Pythia will be used by default.
if COND_PYTHON_SITE
    nodist_python_SCRIPTS = $(PACKAGE).pth
else
    nodist_python_SCRIPTS =
endif
if COND_PYEXEC_SITE
    nodist_pyexec_SCRIPTS = $(PACKAGE).pth
else
    nodist_pyexec_SCRIPTS =
endif
$(PACKAGE).pth:
	echo $(PACKAGE)-$(VERSION) > $@
CLEANFILES = $(PACKAGE).pth

clean-local:
	rm -rf $(top_builddir)/include

## end of Makefile.am
