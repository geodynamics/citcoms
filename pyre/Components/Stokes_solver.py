#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Stokes_solver(Component):


    def __init__(self):
        Component.__init__(self, "stokes-solver", "stokes-solver")
        return


    class Properties(Component.Properties):


        import pyre.properties
        import os

        __properties__ = Component.Properties.__properties__ + (
            
            pyre.properties.string("Solver","cgrad"),
            pyre.properties.bool("node_assemble",True),

            pyre.properties.int("mg_cycle",1),
            pyre.properties.int("down_heavy",1),            
            pyre.properties.int("up_heavy",1),
            pyre.properties.int("vlowstep",2000),            
            pyre.properties.int("vhighstep",3),
            pyre.properties.int("piterations",375),
            pyre.properties.float("accuracy",1.0e-6),
            pyre.properties.float("tole_compressibility",1.0e-7),

            )

# version
__id__ = "$Id: Stokes_solver.py,v 1.1 2003/06/11 23:02:09 tan2 Exp $"

# End of file 
