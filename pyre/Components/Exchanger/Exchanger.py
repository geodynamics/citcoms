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
        #self.selectModule()

        # create c++ exchanger
        #self.createExchanger(solver)
        return



    def selectModule(self):
        import CitcomS.Exchanger
        self.module = CitcomS.Exchanger
        return



    def createExchanger(self, solver):
        self.exchanger = self.module.createExchanger(
                                     solver.communicator,
                                     solver.intercomm,
                                     solver.localLeader,
                                     solver.remoteLeader,
                                     solver.all_variables
                                     )
        return



    def launch(self):
        return





    class Inventory(Component.Inventory):

        import pyre.properties as prop


        inventory = [

            ]



# version
__id__ = "$Id: Exchanger.py,v 1.3 2003/09/05 19:49:14 tan2 Exp $"

# End of file
