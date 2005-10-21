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

dist_bin_SCRIPTS = \
	autocombine.py \
	batchcombine.py \
	batchpaste.sh \
	batchsurf.py \
	combine.py \
	combinesurf.py \
	dxgeneral.sh \
	dxgeneralsurf.sh \
	getcoord.sh \
	getlog.sh \
	getsurf.sh \
	miff2avi \
	miff2mpg \
	parser.py \
	pasteCitcomData.sh \
	zslice.py

visualdir = $(pkgdatadir)/visual
dxmacrodir = $(visualdir)/OpenDXMacro
nobase_dist_visual_DATA = \
	OpenDXMacro/RTFtoXYZ2D.net \
	OpenDXMacro/RTFtoXYZ.net \
	OpenDXMacro/ScaledGlyphs.net \
	OpenDXMacro/ShowSphere.net \
	OpenDXMacro/VelocityRTFtoXYZ2D.net \
	OpenDXMacro/VelocityRTFtoXYZ.net \
	samples/fulltest.cap0.0 \
	samples/fulltest.cap0.0.general \
	samples/fulltest.cap1.0 \
	samples/fulltest.cap10.0 \
	samples/fulltest.cap10.0.general \
	samples/fulltest.cap1.0.general \
	samples/fulltest.cap11.0 \
	samples/fulltest.cap11.0.general \
	samples/fulltest.cap2.0 \
	samples/fulltest.cap2.0.general \
	samples/fulltest.cap3.0 \
	samples/fulltest.cap3.0.general \
	samples/fulltest.cap4.0 \
	samples/fulltest.cap4.0.general \
	samples/fulltest.cap5.0 \
	samples/fulltest.cap5.0.general \
	samples/fulltest.cap6.0 \
	samples/fulltest.cap6.0.general \
	samples/fulltest.cap7.0 \
	samples/fulltest.cap7.0.general \
	samples/fulltest.cap8.0 \
	samples/fulltest.cap8.0.general \
	samples/fulltest.cap9.0 \
	samples/fulltest.cap9.0.general \
	samples/regtest.cap0.0 \
	samples/regtest.cap0.0.general \
	visFull.cfg \
	visRegional.cfg \
	visRegionalSurface.cfg

nodist_visual_DATA = \
	visFull.net \
	visRegional.net \
	visRegionalSurface.net
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
do_subst = sed \
	-e 's|[@]dxmacrodir[@]|$(dxmacrodir)|g'
visFull.net: $(srcdir)/visFull.net.in
	$(do_subst) < $(srcdir)/visFull.net.in > $@ || (rm -f $@ && exit 1)
visRegional.net: $(srcdir)/visRegional.net.in
	$(do_subst) < $(srcdir)/visRegional.net.in > $@ || (rm -f $@ && exit 1)
visRegionalSurface.net: $(srcdir)/visRegionalSurface.net.in
	$(do_subst) < $(srcdir)/visRegionalSurface.net.in > $@ || (rm -f $@ && exit 1)
CitcomSFullImport.net: $(srcdir)/OpenDXMacro/CitcomSFullImport.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSFullImport.net.in > $@ || (rm -f $@ && exit 1)
CitcomSImport.net: $(srcdir)/OpenDXMacro/CitcomSImport.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSImport.net.in > $@ || (rm -f $@ && exit 1)
CitcomSSurfaceHeatflux.net: $(srcdir)/OpenDXMacro/CitcomSSurfaceHeatflux.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSSurfaceHeatflux.net.in > $@ || (rm -f $@ && exit 1)
CitcomSSurfaceImport.net: $(srcdir)/OpenDXMacro/CitcomSSurfaceImport.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSSurfaceImport.net.in > $@ || (rm -f $@ && exit 1)
CitcomSSurfaceTopo.net: $(srcdir)/OpenDXMacro/CitcomSSurfaceTopo.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSSurfaceTopo.net.in > $@ || (rm -f $@ && exit 1)
CitcomSSurfaceVelocity.net: $(srcdir)/OpenDXMacro/CitcomSSurfaceVelocity.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSSurfaceVelocity.net.in > $@ || (rm -f $@ && exit 1)
CitcomSTemperature.net: $(srcdir)/OpenDXMacro/CitcomSTemperature.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSTemperature.net.in > $@ || (rm -f $@ && exit 1)
CitcomSVelocity.net: $(srcdir)/OpenDXMacro/CitcomSVelocity.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSVelocity.net.in > $@ || (rm -f $@ && exit 1)
CitcomSViscosity.net: $(srcdir)/OpenDXMacro/CitcomSViscosity.net.in
	$(do_subst) < $(srcdir)/OpenDXMacro/CitcomSViscosity.net.in > $@ || (rm -f $@ && exit 1)
CLEANFILES = $(nodist_visual_DATA) $(nodist_dxmacro_DATA)
EXTRA_DIST = \
	OpenDXMacro/CitcomSFullImport.net.in \
	OpenDXMacro/CitcomSImport.net.in \
	OpenDXMacro/CitcomSSurfaceHeatflux.net.in \
	OpenDXMacro/CitcomSSurfaceImport.net.in \
	OpenDXMacro/CitcomSSurfaceTopo.net.in \
	OpenDXMacro/CitcomSSurfaceVelocity.net.in \
	OpenDXMacro/CitcomSTemperature.net.in \
	OpenDXMacro/CitcomSVelocity.net.in \
	OpenDXMacro/CitcomSViscosity.net.in \
	visFull.net.in \
	visRegional.net.in \
	visRegionalSurface.net.in

## end of Makefile.am
