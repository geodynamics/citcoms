#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from pyre.components.Component import Component


class Exchanger(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.module = None
        self.exchanger = None
        return



    def initialize(self, solver):
        # choose c++ exchanger module
        self.selectModule()

        # create c++ exchanger
        self.createExchanger(solver)
        return



    def selectModule(self):
        import CitcomS.Exchanger
        self.module = CitcomS.Exchanger
        return



    def createExchanger(self, solver):
        return



    def launch(self):
        return





    class Inventory(Component.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: Exchanger.py,v 1.4 2003/09/08 21:37:42 tan2 Exp $"

# End of file
