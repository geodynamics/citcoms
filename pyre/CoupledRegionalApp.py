#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from RegionalApp import RegionalApp
import Regional as CitcomModule
import journal


class CoupledRegionalApp(RegionalApp):



    class Inventory(RegionalApp.Inventory):

        import pyre.facilities
        from Components.Coupler import Coupler
        from Components.Carrier import Carrier

        inventory = [

            pyre.facilities.facility("coupler",
                                     default=Coupler("coupler", "coupler", CitcomModule)),
            pyre.facilities.facility("carrier",
                                     default=Carrier("carrier", "carrier", CitcomModule)),

            ]


# version
__id__ = "$Id: CoupledRegionalApp.py,v 1.1 2003/08/22 22:18:41 tan2 Exp $"

# End of file
