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


class FullApp(Application):


    def run(self):

        full = self.inventory.citcom
        full.run()

        return



    class Inventory(Application.Inventory):

        import pyre.facilities
        from CitcomSFull import CitcomSFull

        inventory = [

            pyre.facilities.facility("citcom", default=CitcomSFull()),

            ]

# version
__id__ = "$Id: FullApp.py,v 1.4 2003/08/25 19:34:40 tan2 Exp $"

# End of file
