## Process this file with automake to produce Makefile.in
# $Id: Makefile.am,v 1.1 2005/09/09 16:12:04 leif Exp $

subpackage = merlin
include $(top_srcdir)/packages/subpackage.am

SUBDIRS = etc

# applications
dist_bin_SCRIPTS = applications/merlin

# merlin
nobase_pkgpython_PYTHON = \
	merlin/agents/Agent.py \
	merlin/agents/Builder.py \
	merlin/agents/CodeBuilder.py \
	merlin/agents/__init__.py \
	merlin/agents/Janitor.py \
	merlin/agents/Librarian.py \
	merlin/agents/ProjectInspector.py \
	merlin/agents/ProjectVisualizer.py \
	merlin/assets/Archive.py \
	merlin/assets/AssetContainer.py \
	merlin/assets/Asset.py \
	merlin/assets/Executable.py \
	merlin/assets/FileContainer.py \
	merlin/assets/Header.py \
	merlin/assets/Headers.py \
	merlin/assets/__init__.py \
	merlin/assets/Library.py \
	merlin/assets/Object.py \
	merlin/assets/Project.py \
	merlin/assets/PythonPackage.py \
	merlin/assets/SharedObject.py \
	merlin/assets/Source.py \
	merlin/assets/Sources.py \
	merlin/components/Curator.py \
	merlin/components/__init__.py \
	merlin/components/Merlin.py \
	merlin/__init__.py \
	merlin/projects/Application.py \
	merlin/projects/__init__.py \
	merlin/projects/python/__init__.py \
	merlin/projects/python/Package.py

# tests
EXTRA_DIST = \
	tests/hello/hello.py \
	tests/hello/__init__.py \
	tests/hello/.merlin \
	tests/libhello/hello.cc \
	tests/libhello/hello.h \
	tests/libhello/.merlin

## end of Makefile.am
