#!/usr/bin/env python
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.simulations.Solver import Solver


class Rigid(Solver):


    class Inventory(Solver.Inventory):

        import pyre.inventory
        from pyre.units.SI import second

        syncOnInit = pyre.inventory.bool("syncOnInit", default=True)
        timestep = pyre.inventory.dimensional("timestep", default=1.0e-6 * second)


    def launch(self, application):
        Solver.launch(self, application)

        # verify the machine layout
        layout = application.layout  
        rank = layout.rank
        communicator = layout.communicator

        if not communicator:
            import journal
            journal.error(self.name).log("null communicator")
            return

        if communicator.size > 1:
            import journal
            journal.error(self.name).log("this is a single processor solver")
            return

        # save the communicator info
        self._solidServer = rank
        self._fluidServer = (rank + 1) % 2

        # register the solver coupler
        self.coupler = application.coupler

        # create the model
        self.initializeModel(application)

        # initial boundary synchronization with the fluid
        if self.inventory.syncOnInit:
            self.applyBoundaryConditions()
            
        from pyre.units.SI import second
        t, step = 0.0*second, 0

        return (t, step)


    def initializeModel(self, application):
        # create the model
        model = application.modeller.model()

        # mesh its surface
        mesher = application.surfaceMesher
        mesher.inventory.gridAspectRatio = 1
        mesh, bbox = mesher.facet(model)

        # record this as the surface mesh in the coupler's boundary data structure
        application.coupler.setMesh(mesh)

        return


    def visualize(self):
        self.publishState("")
        return


    def applyBoundaryConditions(self):
        Solver.applyBoundaryConditions(self)
        self.coupler.exchangeBoundary()
        self.coupler.exchangeFields()
        return


    def stableTimestep(self):
        dt = self.inventory.timestep
    
        sink = self._fluidServer
        source = self._solidServer

        from rigid import timestep
        from pyre.units.time import second
        dt = timestep(sink, source, dt.value) * second

        Solver.stableTimestep(self, dt)
        return dt


    def publishState(self, directory):
        Solver.publishState(self, directory)

        # pRange = self.coupler.pressureRange()
        # self._monitorInfo.log("pressure range: (%g, %g) pascal" % pRange)

        # NYI
        # self.coupler.publish()

        Solver.publishState(self, directory)
        return


    def verifyInterface(self):
        import rigid
        # NYI
        # return rigid.verify(self.coupler.boundary())


    def __init__(self):
        Solver.__init__(self, "rigid")

        self.coupler = None
        self._fluidServer = None
        self._solidServer = None

        return


# version
__id__ = "$Id: Rigid.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

#
# End of file
