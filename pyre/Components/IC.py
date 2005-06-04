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
        inv.blob_center = map(float, inv.blob_center)

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


        import pyre.inventory



        restart = pyre.inventory.bool("restart", default=False)
        post_p = pyre.inventory.bool("post_p", default=False)
        solution_cycles_init = pyre.inventory.int("solution_cycles_init", default=0)
        zero_elapsed_time = pyre.inventory.bool("zero_elapsed_time", default=True)

        tic_method = pyre.inventory.int("tic_method", default=0,
                            validator=pyre.inventory.choice([0, 1, 2]))

        half_space_age = pyre.inventory.float("half_space_age", default=40,
                              validator=pyre.inventory.greater(1e-3))

        num_perturbations = pyre.inventory.int("num_perturbations", default=1,
                            validator=pyre.inventory.less(255))
        perturbmag = pyre.inventory.list("perturbmag", default=[0.05])
        perturbl = pyre.inventory.list("perturbl", default=[1])
        perturbm = pyre.inventory.list("perturbm", default=[1])
        perturblayer = pyre.inventory.slice("perturblayer", default=[5])


        blob_center = pyre.inventory.list("blob_center", default=[-999., -999., -999.])
        blob_radius = pyre.inventory.float("blob_radius", default=[0.063])
        blob_dT = pyre.inventory.float("blob_dT", default=[0.18])


# version
__id__ = "$Id: IC.py,v 1.16 2005/06/03 21:51:43 leif Exp $"

# End of file
