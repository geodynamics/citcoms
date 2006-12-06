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

nobase_pkgpython_PYTHON = \
	__init__.py \
	BaseApplication.py \
	Components/__init__.py \
	Components/Advection_diffusion/Advection_diffusion.py \
	Components/Advection_diffusion/__init__.py \
	Components/BC.py \
	Components/CitcomComponent.py \
	Components/Const.py \
	Components/IC.py \
	Components/Output.py \
	Components/Param.py \
	Components/Phase.py \
	Components/Sphere/__init__.py \
	Components/Sphere/FullSphere.py \
	Components/Sphere/RegionalSphere.py \
	Components/Sphere/Sphere.py \
	Components/Stokes_solver/Incompressible.py \
	Components/Stokes_solver/__init__.py \
	Components/Tracer.py \
	Components/Visc.py \
	Controller.py \
	CoupledApp.py \
	Coupler/__init__.py \
	Coupler/ContainingCoupler.py \
	Coupler/Coupler.py \
	Coupler/EmbeddedCoupler.py \
	Coupler/Inlet.py \
	Coupler/Outlet.py \
	Layout.py \
	SimpleApp.py \
	Solver/__init__.py \
	Solver/CoupledFullSolver.py \
	Solver/CoupledRegionalSolver.py \
	Solver/CoupledSolver.py \
	Solver/FullSolver.py \
	Solver/RegionalSolver.py \
	Solver/Solver.py

nodist_pkgpython_PYTHON = \
	config.py
config.py: $(top_builddir)/config.h $(top_builddir)/Makefile
	$(PYCONFIG) -h $(top_builddir)/config.h -m $(top_builddir)/Makefile -o $@
CLEANFILES = $(nodist_pkgpython_PYTHON)

## end of Makefile.am
