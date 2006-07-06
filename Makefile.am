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

ACLOCAL_AMFLAGS = -I ./m4

if COND_PYRE
    MAYBE_PYRE = CitcomS etc examples module tests visual
endif

SUBDIRS = $(MAYBE_PYTHIA) $(MAYBE_EXCHANGER) lib $(MAYBE_PYRE) bin
DIST_SUBDIRS = \
	$(MAYBE_DIST_PYTHIA) $(MAYBE_DIST_EXCHANGER) \
	lib bin \
	CitcomS etc examples module tests visual

EXTRA_DIST = \
	doc/CitComS.doxyconf

## end of Makefile.am
