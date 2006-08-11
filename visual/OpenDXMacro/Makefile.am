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

SUFFIXES = .net.in .net

visualdir = $(pkgdatadir)/visual
dxmacrodir = $(visualdir)/OpenDXMacro
abs_builddir = @abs_builddir@

dist_dxmacro_DATA = \
	RTFtoXYZ2D.net \
	RTFtoXYZ.net \
	ScaledGlyphs.net \
	ShowSphere.net \
	VelocityRTFtoXYZ2D.net \
	VelocityRTFtoXYZ.net

nodist_dxmacro_DATA = \
	CitcomSFullImport.net \
	CitcomSImport.net \
	CitcomSSurfaceHeatflux.net \
	CitcomSSurfaceImport.net \
	CitcomSSurfaceTopo.net \
	CitcomSSurfaceVelocity.net \
	CitcomSTemperature.net \
	CitcomSVelocity.net \
	CitcomSViscosity.net

dxmacro_templates = \
	CitcomSFullImport.net.in \
	CitcomSImport.net.in \
	CitcomSSurfaceHeatflux.net.in \
	CitcomSSurfaceImport.net.in \
	CitcomSSurfaceTopo.net.in \
	CitcomSSurfaceVelocity.net.in \
	CitcomSTemperature.net.in \
	CitcomSVelocity.net.in \
	CitcomSViscosity.net.in

do_build = sed -e s:[@]dxmacrodir[@]:$(abs_builddir):g
do_install = sed -e s:[@]dxmacrodir[@]:$(dxmacrodir):g

.net.in.net:
	$(do_build) $< > $@ || (rm -f $@ && exit 1)

install-nodist_dxmacroDATA: $(nodist_dxmacro_DATA)
	@$(NORMAL_INSTALL)
	test -z "$(dxmacrodir)" || $(mkdir_p) "$(DESTDIR)$(dxmacrodir)"
	@list='$(nodist_dxmacro_DATA)'; for p in $$list; do \
	  if test -f "$$p"; then d=; else d="$(srcdir)/"; fi; \
	  f=$(am__strip_dir) \
	  echo " $(do_install) '$$d$$p.in' '$(DESTDIR)$(dxmacrodir)/$$f'"; \
	  $(do_install) "$$d$$p.in" > "$(DESTDIR)$(dxmacrodir)/$$f"; \
	  echo " chmod 644 '$(DESTDIR)$(dxmacrodir)/$$f'"; \
	  chmod 644 "$(DESTDIR)$(dxmacrodir)/$$f"; \
	done

CLEANFILES = $(nodist_dxmacro_DATA)
EXTRA_DIST = $(dxmacro_templates)

## end of Makefile.am
