#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from CitcomComponent import CitcomComponent

class IC(CitcomComponent):


    def __init__(self, name="ic", facility="ic"):
        CitcomComponent.__init__(self, name, facility)
        return



    def setProperties(self):
        self.CitcomModule.IC_set_properties(self.all_variables, self.inventory)
        return



    def launch(self):
        self.initTemperature()
        self.initPressure()
        self.initVelocity()
        self.initViscosity()
        return



    def initTemperature(self):
        if self.inventory.restart:
            self.CitcomModule.restartTemperature(self.all_variables)
        else:
            self.CitcomModule.constructTemperature(self.all_variables)
        return



    def initPressure(self):
        self.CitcomModule.initPressure(self.all_variables)
        return



    def initVelocity(self):
        self.CitcomModule.initVelocity(self.all_variables)
        return



    def initViscosity(self):
        self.CitcomModule.initViscosity(self.all_variables)
        return



    class Inventory(CitcomComponent.Inventory):


        import pyre.properties


        inventory = [

            pyre.properties.bool("restart", default=False),
            pyre.properties.bool("post_p", default=False),
            pyre.properties.int("solution_cycles_init", default=0),
            pyre.properties.bool("zero_elapsed_time", default=True),

            pyre.properties.int("num_perturbations",2),
            pyre.properties.sequence("perturbmag",[0.05,0.05]),
            pyre.properties.sequence("perturbl",[2,2]),
            pyre.properties.sequence("perturbm",[2,2]),
            pyre.properties.sequence("perturblayer",[3,6]),

            ]

# version
__id__ = "$Id: IC.py,v 1.9 2003/11/28 22:17:26 tan2 Exp $"

# End of file
