#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import journal


class RegionalApp(Application):


    def run(self):

        regional = self.inventory.citcom
        regional.run()

        return



    class Inventory(Application.Inventory):

        import pyre.facilities
        from CitcomSRegional import CitcomSRegional

        inventory = [

            pyre.facilities.facility("citcom", default=CitcomSRegional()),

            ]



# version
__id__ = "$Id: RegionalApp.py,v 1.28 2003/08/25 19:16:04 tan2 Exp $"

# End of file
