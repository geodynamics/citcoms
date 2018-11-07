#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CoupledSolver import CoupledSolver

class CoupledRegionalSolver(CoupledSolver):


    def initializeSolver(self):
        from CitcomSLib import regional_solver_init
        # define common aliases for full/regional functions
        regional_solver_init(self.all_variables)



    class Inventory(CoupledSolver.Inventory):

        import pyre.inventory

        # component modules
        import CitcomS.Components.Sphere as Sphere


        mesher = pyre.inventory.facility("mesher", factory=Sphere.regionalSphere, args=("regional-sphere",))

        datafile = pyre.inventory.str("datafile", default="regtest")
        datafile_old = pyre.inventory.str("datafile_old", default="regtest")




# version
__id__ = "$Id$"

# End of file
