# -*- Makefile -*-
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

include local.def
TYPE=Regional

PROJECT = CitcomS
PACKAGE = $(TYPE)module
include std-pythonmodule.def

EXTERNAL_INCLUDES += $(PYTHIA_INCDIR)
EXTERNAL_LIBDIRS = $(PYTHIA_LIBDIR)
EXTERNAL_LIBPATH += $(foreach dir,$(EXTERNAL_LIBDIRS),-L$(dir))
ifeq (Linux,$(findstring Linux,$(PLATFORM_ID)))
RPATH_ARGS = $(foreach dir,$(PROJ_LIBDIR) $(EXTERNAL_LIBDIRS),-Xlinker -rpath $(dir))
else
RPATH_ARGS =
endif

PROJ_CXX_SRCLIB = \
	-l$(PROJECT)Common \
	-l$(PROJECT)$(TYPE) \
	-l_mpimodule \
	-ljournal \
	$(RPATH_ARGS)

PROJ_CXX_INCLUDES = ../../lib/Common

PROJ_SRCS = \
    advdiffu.cc \
    bindings.cc \
    exceptions.cc \
    initial_conditions.cc \
    mesher.cc \
    misc.cc \
    outputs.cc \
    setProperties.cc \
    stokes_solver.cc

# version
# $Id$

# End of file
