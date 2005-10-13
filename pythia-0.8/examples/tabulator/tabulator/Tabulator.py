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

from pyre.components.Component import Component


class Tabulator(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        low = pyre.inventory.float("low", default=0.0)
        high = pyre.inventory.float("high", default=1.0)
        step = pyre.inventory.float("step", default=0.1)


    def tabulate(self):
        import tabulator._tabulator
        functor = self.functor
        functor.initialize()
        
        tabulator._tabulator.tabulate(
            self.low, self.high, self.step, functor.handle)

        return


    def __init__(self):
        Component.__init__(self, "tabulator", "tabulator")
        self.low = 0.0
        self.high = 1.0
        self.step = 0.1
        self.functor = None
        return


    def _init(self):
        self.low = self.inventory.low
        self.high = self.inventory.high
        self.step = self.inventory.step
        return


# version
__id__ = "$Id: Tabulator.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

# End of file 
