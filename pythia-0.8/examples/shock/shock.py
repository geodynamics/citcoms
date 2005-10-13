#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from mpi.Application import Application


class ShockApp(Application):


    class Inventory(Application.Inventory):

        import pyre.inventory

        steps = pyre.inventory.int("steps", default=10)

        # geometry
        modeller = pyre.inventory.facility("modeller", default="cube")

        # surface mesher
        import acis
        surfaceMesher = pyre.inventory.facility("surfaceMesher", factory=acis.surfaceMesher)

        # machine management
        layout = pyre.inventory.facility("layout", default="coupled")

        # simulation control
        import pyre.simulations
        controller = pyre.inventory.facility("controller", factory=pyre.simulations.controller)

        # solvers
        import rigid
        solid = pyre.inventory.facility('solid', family='solver', factory=rigid.solver)

        import pulse
        fluid = pyre.inventory.facility('fluid', family='solver', factory=pulse.solver)

        import elc
        coupler = pyre.inventory.facility('coupler', factory=elc.mpiExchanger)


    def main(self, *args, **kwds):
        # configure the parallel machine
        self.layout.layout(self)

        # print some information
        self.reportConfiguration()

        # initialize the coupler
        self.coupler.initialize(self) # uses the world communicator for the exchange by default

        # launch the application
        self.controller.solver = self.layout.solver
        self.controller.launch(self)

        # compute the specified number of steps
        self.controller.march(steps=self.inventory.steps)

        return


    def reportConfiguration(self):
        if self.layout.rank == 0:
            import journal
            # journal.debug("elc.memory").activate()
            # journal.debug("elc.exchange").activate()
            # journal.debug("pulse.generators").activate()
        elif self.layout.rank == 1:
            import journal
            # journal.debug("elc.memory").activate()
            # journal.debug("elc.exchange").activate()
            # journal.debug("pulse.generators").activate()
    
        # journal.debug("rigid.monitoring").activate()
        # journal.debug("rigid.timeloop").activate()
        # journal.debug("pulse.monitoring").activate()
        # journal.debug("pulse.timeloop").activate()

        self.fluid.dump()

        return
        

    def __init__(self):
        Application.__init__(self, 'shock')
        self.modeller = None
        self.surfaceMesher = None
        self.layout = None
        self.controller = None
        self.fluid = None
        self.solid = None
        self.coupler = None
        return


    def _defaults(self):
        Application._defaults(self)
        self.inventory.launcher.inventory.nodes = 2
        return


    def _configure(self):
        Application._configure(self)
        self.modeller = self.inventory.modeller
        self.surfaceMesher = self.inventory.surfaceMesher
        self.layout = self.inventory.layout
        self.controller = self.inventory.controller
        self.fluid = self.inventory.fluid
        self.solid = self.inventory.solid
        self.coupler = self.inventory.coupler
        return


    def _init(self):
        Application._init(self)
        return


# main
if __name__ == '__main__':
    app = ShockApp()
    app.run()


# version
__id__ = "$Id: shock.py,v 1.1.1.1 2005/03/08 16:14:00 aivazis Exp $"

# End of file 
