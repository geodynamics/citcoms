## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/20 23:33:50 leif Exp $

# In creating mpipython.exe, we are simply re-linking the Python
# interpreter with MPI.  To this end, re-create the 'python' link line
# using Python's own 'Makefile.pre.in' and 'configure.in' as a
# reference.  Via distutils, our 'configure' script obtains the actual
# 'make' variable values used during the Python install.  Note that
# mpipython.exe's main() is always a C++ MPI program, whereas Python's
# main() could be plain old C; therefore PYTHON_LINKCC is set to
# MPICXX (usually mpicxx/mpic++) instead of the link command used to
# build 'python'.

CXX = $(MPICXX)
EXEEXT = .exe
INCLUDES = -I$(PYTHON_INCDIR) $(MPIINCLUDES)
bin_PROGRAMS = mpipython
PYTHON_LINKCC = $(MPICXX)
PYTHON_MAINOBJ = $(mpipython_OBJECTS)
mpipython_SOURCES = mpipython.cc dummy.cc
mpipython_LDFLAGS = $(PYTHON_LDFLAGS) $(PYTHON_LINKFORSHARED)

mpipython$(EXEEXT): $(PYTHON_MAINOBJ) $(mpipython_DEPENDENCIES) 
	@rm -f mpipython$(EXEEXT)
	$(PYTHON_LINKCC) $(PYTHON_LDFLAGS) $(PYTHON_LINKFORSHARED) -o $@ \
		$(PYTHON_MAINOBJ) \
		$(PYTHON_BLDLIBRARY) \
		$(MPILIBS) \
		$(PYTHON_LIBS) $(PYTHON_MODLIBS) $(PYTHON_SYSLIBS) $(PYTHON_LDLAST)

## end of Makefile.am
