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
        inv = self.inventory
        inv.perturbmag = map(float, inv.perturbmag)
        inv.perturbl = map(float, inv.perturbl)
        inv.perturbm = map(float, inv.perturbm)

        self.CitcomModule.IC_set_properties(self.all_variables, inv)
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

            pyre.properties.int("tic_method", default=0,
                                validator=pyre.properties.choice([0, 1])),

            pyre.properties.float("half_space_age", default=40,
                                  validator=pyre.properties.greater(1e-3)),

            pyre.properties.int("num_perturbations", default=1,
                                validator=pyre.properties.less(255)),
            pyre.properties.list("perturbmag", default=[0.05]),
            pyre.properties.list("perturbl", default=[1]),
            pyre.properties.list("perturbm", default=[1]),
            pyre.properties.sequence("perturblayer", default=[5]),

            ]

# version
__id__ = "$Id: IC.py,v 1.13 2005/02/18 00:13:36 tan2 Exp $"

# End of file
