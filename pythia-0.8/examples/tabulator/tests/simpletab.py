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


class SimpleApp(Script):


    class Inventory(Script.Inventory):

        import tabulator
        import pyre.inventory

        tabulator = pyre.inventory.facility("tabulator", factory=tabulator.simple)


    def main(self, *args, **kwds):
	simple = self.inventory.tabulator
        simple.tabulate()
            
        return


    def __init__(self):
        Script.__init__(self, "simpletab")
        return


# main

if __name__ == "__main__":
    app = SimpleApp()
    app.run()

# version
__id__ = "$Id: simpletab.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

# End of file 
