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

from pyre.applications.Script import Script


class SimulationTest(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory
        import pyre.simulations

        solver = pyre.inventory.facility(
            "solver", default=pyre.simulations.simpleSolver("solver"))

        controller = pyre.inventory.facility(
            "controller", default=pyre.simulations.controller("controller"))


    def main(self):
        self.reportConfiguration()

        solver = self.inventory.solver
        controller = self.inventory.controller

        controller.solver = solver
        controller.launch(self)

        controller.march(steps=10)

        return


    def reportConfiguration(self):
        self._info.line("configuration:")
        self._info.line("  properties:")
        self._info.line("    name: %r" % self.name)
        self._info.line("    full name: %r" % self.filename)

        self._info.line("  facilities:")
        self._info.line("    journal: %r" % self.inventory.journal.name)
        self._info.line("    controller: %r" % self.inventory.controller.name)
        self._info.line("    solver: %r" % self.inventory.solver.name)
        self._info.line("")

        return


    def __init__(self):
        Script.__init__(self, "simulation")
        return


# main

if __name__ == "__main__":
    import journal
    journal.info("simulation").activate()
    journal.debug("simulation").activate()

    journal.info("solver.timeloop").activate()
    journal.debug("solver.timeloop").activate()
    
    app = SimulationTest()
    app.run()


# version
__id__ = "$Id: simulation.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

# End of file 
