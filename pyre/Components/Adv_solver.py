#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Adv_solver(Component):


    def __init__(self):
        Component.__init__(self, "adv-solver", "adv-solver")
        return


    class Properties(Component.Properties):


        import pyre.properties
        import os

        __properties__ = Component.Properties.__properties__ + (
            
            pyre.properties.bool("ADV",True),
            pyre.properties.float("fixed_timestep",0.0),
            pyre.properties.float("finetunedt",0.7),

            pyre.properties.int("adv_sub_iterations",2),
            pyre.properties.float("maxadvtime",10.0),
            pyre.properties.bool("precond",True),

            pyre.properties.bool("aug_lagr",True),
            pyre.properties.float("aug_number",2.0e3),

            )

# version
__id__ = "$Id: Adv_solver.py,v 1.1 2003/06/11 23:02:09 tan2 Exp $"

# End of file 
