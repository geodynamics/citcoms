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

from CitcomComponent import CitcomComponent

class Phase(CitcomComponent):


    def __init__(self, name="phase", facility="phase"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self, stream):
        from CitcomSLib import Phase_set_properties
        Phase_set_properties(self.all_variables, self.inventory, stream)
        return


    class Inventory(CitcomComponent.Inventory):


        import pyre.inventory


        Ra_410 = pyre.inventory.float("Ra_410", default=0.0)
        clapeyron410 = pyre.inventory.float("clapeyron410", default=0.0235)
        transT410 = pyre.inventory.float("transT410", default=0.78)
        width410 = pyre.inventory.float("width410", default=0.0058)

        Ra_670 = pyre.inventory.float("Ra_670", default=0.0)
        clapeyron670 = pyre.inventory.float("clapeyron670", default=-0.0235)
        transT670 = pyre.inventory.float("transT670", default=0.78)
        width670 = pyre.inventory.float("width670", default=0.0058)

        Ra_cmb = pyre.inventory.float("Ra_cmb", default=0.0)
        clapeyroncmb = pyre.inventory.float("clapeyroncmb", default=-0.0235)
        transTcmb = pyre.inventory.float("transTcmb", default=0.875)
        widthcmb = pyre.inventory.float("widthcmb", default=0.0058)


# version
__id__ = "$Id$"

# End of file
