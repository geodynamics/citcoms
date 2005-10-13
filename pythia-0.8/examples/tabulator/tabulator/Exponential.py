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


class Exponential(Component):


    class Inventory(Component.Inventory):
        
        import pyre.inventory

        a = pyre.inventory.float("a", default=0.0)


    def initialize(self):
        self.a = self.inventory.a
        
        import tabulator._tabulator
        tabulator._tabulator.exponentialSet(self.a)
        return


    def __init__(self):
        Component.__init__(self, "exponential", "functor")
        self.a = 0.0

        import tabulator._tabulator
        self.handle = tabulator._tabulator.exponential()

        return


    def _init(self):
        Component._init(self)
        self.initialize()
        return


# version
__id__ = "$Id: Exponential.py,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $"

# End of file 
