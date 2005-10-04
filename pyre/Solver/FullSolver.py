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

from Solver import Solver
import journal


class FullSolver(Solver):


    def __init__(self, name, facility="solver"):
	Solver.__init__(self, name, facility)
        try:
            import CitcomS.Full as CitcomModule
            self.CitcomModule = CitcomModule
        except:
            self.CitcomModule = None
        return



    class Inventory(Solver.Inventory):

        import pyre.inventory

        # component modules
        import CitcomS.Components.Sphere as Sphere


        mesher = pyre.inventory.facility("mesher", factory=Sphere.fullSphere, args=("full-sphere",))

        datafile = pyre.inventory.str("datafile", default="fulltest")
        datafile_old = pyre.inventory.str("datafile_old", default="fulltest")




# version
__id__ = "$Id$"

# End of file
