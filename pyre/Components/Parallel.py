#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component

class Parallel(Component):


    def __init__(self):
        Component.__init__(self, "parallel", "parallel")
        return


    class Properties(Component.Properties):


        import pyre.properties

        __properties__ = Component.Properties.__properties__ + (
            pyre.properties.list("nproc_surf",1,
				 pyre.properties.choice([1,12])
				 ),            
            pyre.properties.int("nprocx",1),
            pyre.properties.int("nprocy",1),
            pyre.properties.int("nprocz",1),            
            # debugging flags?
            )


# version
__id__ = "$Id: Parallel.py,v 1.1 2003/06/11 23:02:09 tan2 Exp $"

# End of file 
