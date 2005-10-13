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


class Quadratic(Component):


    class Inventory(Component.Inventory):
        
        import pyre.inventory

        a = pyre.inventory.float("a", default=0.0)
        b = pyre.inventory.float("b", default=0.0)
        c = pyre.inventory.float("c", default=0.0)


    def initialize(self):
        self.a = self.inventory.a
        self.b = self.inventory.b
        self.c = self.inventory.c

        import tabulator._tabulator
        tabulator._tabulator.quadraticSet(self.a, self.b, self.c)
        return


    def __init__(self):
        Component.__init__(self, "quadratic", "functor")
        self.a = 0.0
        self.b = 0.0
        self.c = 0.0

        import tabulator._tabulator
        self.handle = tabulator._tabulator.quadratic()
        
        return


    def _init(self):
        Component._init(self)
        self.initialize()
        return


# version
__id__ = "$Id: Quadratic.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

# End of file 
