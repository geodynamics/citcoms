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


    class Properties(Component.Properties):


        import pyre.properties
        import os

        __properties__ = Component.Properties.__properties__ + (
            pyre.properties.int("num_perturbations",2),
            pyre.properties.sequence("perturbmag",[0.05,0.05]),
            pyre.properties.sequence("perturbl",[2,2]),
            pyre.properties.sequence("perturbm",[2,2]),
            pyre.properties.sequence("perturblayer",[3,6]),
            )

# version
__id__ = "$Id: IC.py,v 1.1 2003/06/11 23:02:09 tan2 Exp $"

# End of file 
