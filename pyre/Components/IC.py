#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class IC(Component):


    def __init__(self):
        Component.__init__(self, "ic", "ic")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.IC_set_prop(self.inventory)
        return



    class Inventory(Component.Inventory):


        import pyre.properties


        __inventory__ = [

            pyre.properties.int("num_perturbations",2),
            pyre.properties.sequence("perturbmag",[0.05,0.05]),
            pyre.properties.sequence("perturbl",[2,2]),
            pyre.properties.sequence("perturbm",[2,2]),
            pyre.properties.sequence("perturblayer",[3,6]),

            ]

# version
__id__ = "$Id: IC.py,v 1.2 2003/07/09 19:42:27 tan2 Exp $"

# End of file
