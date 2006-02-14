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


class Launcher(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        nodes = pyre.inventory.int("nodes", default=0)
        nodelist = pyre.inventory.slice("nodelist")


    def launch(self):
        raise NotImplementedError("class '%s' must override 'launch'" % self.__class__.__name__)


    def __init__(self, name):
        Component.__init__(self, name, facility="launcher")
        self.nodes = 0
        self.nodelist = None
        return


    def _configure(self):
        self.nodes = self.inventory.nodes
        self.nodelist = self.inventory.nodelist
        return


# version
__id__ = "$Id: Launcher.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file 
