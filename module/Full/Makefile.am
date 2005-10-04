## Process this file with automake to produce Makefile.in
##
##<LicenseText>
##
## CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
## Copyright (C) 2002-2005, California Institute of Technology.
##
## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program; if not, write to the Free Software
## Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
##
##</LicenseText>

# $Id$

# Fullmodule is identical to Regionalmodule, except that it is linked
# against libCitcomSFull instead of libCitcomSRegional.

pkgpyexec_LTLIBRARIES = Fullmodule.la
INCLUDES = -I$(top_srcdir)/module/Regional -I$(PYTHON_INCDIR)
Fullmodule_la_SOURCES = Fullmodule.cc
Fullmodule_la_LDFLAGS = -module -release $(VERSION)
Fullmodule_la_LIBADD = \
	$(top_builddir)/module/Regional/libModuleCommon.la \
	$(top_builddir)/lib/Full/libCitcomSFull.la \
	-l_mpimodule -ljournal

## end of Makefile.am
