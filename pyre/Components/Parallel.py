#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class Parallel(CitcomComponent):


    class Inventory(CitcomComponent.Inventory):


        import pyre.properties

        inventory = [

            pyre.properties.int("nproc_surf",1,
                                pyre.properties.choice([1,12])),
            pyre.properties.int("nprocx",1),
            pyre.properties.int("nprocy",1),
            pyre.properties.int("nprocz",1),

            ]


# version
__id__ = "$Id: Parallel.py,v 1.5 2003/07/24 17:46:46 tan2 Exp $"

# End of file
