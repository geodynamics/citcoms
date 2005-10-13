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


class Simple(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        a = pyre.inventory.float("a", default=0.0)
        low = pyre.inventory.float("low", default=0.0)
        high = pyre.inventory.float("high", default=1.0)
        step = pyre.inventory.float("step", default=0.1)


    def tabulate(self):
        import tabulator._tabulator
        
        tabulator._tabulator.simpletab(self.a, self.low, self.high, self.step)

        return


    def __init__(self):
        Component.__init__(self, "tabulator", "tabulator")
        self.a = 0.0
        self.low = 0.0
        self.high = 1.0
        self.step = 0.1
        self.functor = None
        return


    def _init(self):
        self.a = self.inventory.a
        self.low = self.inventory.low
        self.high = self.inventory.high
        self.step = self.inventory.step
        return


# version
__id__ = "$Id: Simple.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

# End of file 
