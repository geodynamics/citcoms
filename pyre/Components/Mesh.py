#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Mesh(Component):


    def __init__(self):
        Component.__init__(self, "mesh", "mesh")
        return



    def setProperties(self):
        import CitcomS.Regional as Regional
	Regional.Mesh_set_prop(self.inventory)
        return



    class Inventory(Component.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.bool("coord",False),
            pyre.properties.str("coord_file","coord.dat"),

            pyre.properties.int("nodex",17),
            pyre.properties.int("nodey",17),
            pyre.properties.int("nodez",9),
            pyre.properties.int("mgunitx",8),
            pyre.properties.int("mgunity",8),
            pyre.properties.int("mgunitz",8),
            pyre.properties.int("levels",3),

            pyre.properties.float("theta_min",1.5708),
            pyre.properties.float("theta_max",2.79523),
            pyre.properties.float("phi_min",2.26893),
            pyre.properties.float("phi_max",3.83972),
            pyre.properties.float("radius_inner",0.55),
            pyre.properties.float("radius_outer",1.0),

            ]


# version
__id__ = "$Id: Mesh.py,v 1.2 2003/07/09 19:42:27 tan2 Exp $"

# End of file
