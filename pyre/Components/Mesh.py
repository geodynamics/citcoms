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


    class Properties(Component.Properties):


        import pyre.properties


        __properties__ = Component.Properties.__properties__ + (

            pyre.properties.bool("coord",False),
            pyre.properties.string("coord_file","coord.dat"),
            
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
            
            )


# version
__id__ = "$Id: Mesh.py,v 1.1 2003/06/11 23:02:09 tan2 Exp $"

# End of file 
