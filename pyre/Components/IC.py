#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class IC(CitcomComponent):


    def setProperties(self):
        self.CitcomModule.IC_set_properties(self.all_variables, self.inventory)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.int("num_perturbations",2),
            pyre.properties.sequence("perturbmag",[0.05,0.05]),
            pyre.properties.sequence("perturbl",[2,2]),
            pyre.properties.sequence("perturbm",[2,2]),
            pyre.properties.sequence("perturblayer",[3,6]),

            ]

# version
__id__ = "$Id: IC.py,v 1.6 2003/08/27 20:52:47 tan2 Exp $"

# End of file
