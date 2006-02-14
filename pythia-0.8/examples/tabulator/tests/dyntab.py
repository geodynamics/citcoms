#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.applications.Script import Script


class DyntabApp(Script):


    class Inventory(Script.Inventory):

        import tabulator
        import pyre.inventory

        functor = pyre.inventory.facility("functor", factory=tabulator.exponential)
        tabulator = pyre.inventory.facility("tabulator", factory=tabulator.tabulator)

    
    def main(self, *args, **kwds):
        import tabulator
        functor = self.inventory.functor
        tabulator = self.inventory.tabulator

        tabulator.functor = functor

        tabulator.tabulate()
            
        return


    def __init__(self):
        Script.__init__(self, "dyntab")
        return


# main

if __name__ == "__main__":
    app = DyntabApp()
    app.run()

# version
__id__ = "$Id: dyntab.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

# End of file 
