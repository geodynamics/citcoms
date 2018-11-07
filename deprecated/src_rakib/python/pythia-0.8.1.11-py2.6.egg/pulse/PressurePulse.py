#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from pyre.simulations.Solver import Solver


class PressurePulse(Solver):


    class Inventory(Solver.Inventory):

        import pyre.inventory
        from pyre.units.SI import second
        from HeavisidePulse import HeavisidePulse

        syncOnInit = pyre.inventory.bool("syncOnInit", default=True)
        generator = pyre.inventory.facility("generator", factory=HeavisidePulse)
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
        self._fluidServer = rank
        self._solidServer = (rank + 1) % 2

        self.coupler = application.coupler

        # initial boundary synchronization with the fluid
        if self.inventory.syncOnInit:
            self.applyBoundaryConditions()
        else:
            raise "pulse requires options.syncBoundaryInitialization=true"
            
        from pyre.units.SI import second
        t, step = 0.0*second, 0

        return (t, step)


    def applyBoundaryConditions(self):
        Solver.applyBoundaryConditions(self)
    
        import mpi
        rank = mpi.world().rank

        self.coupler.exchangeBoundary()
        self.generator.updatePressure(self.coupler.boundary)
        self.coupler.exchangeFields()
    
        return

    
    def stableTimestep(self):
        dt = self.inventory.timestep
    
        sink = self._fluidServer
        source = self._solidServer
        
        import pulse
        from pyre.units.time import second
        dt = pulse.timestep(sink, source, dt.value) * second

        Solver.stableTimestep(self, dt)
        return dt


    def publishState(self, directory):
        # NYI
        # self.coupler.publish()
        Solver.publishState(self, directory)
        return


    def advance(self, dt):
        Solver.advance(self, dt)
        self.inventory.generator.advance(dt)
        return


    def verifyInterface(self):
        import pulse
        # NYI
        # return pulse.verify(self.coupler.boundary())


    def __init__(self):
        Solver.__init__(self, "pulse")

        self.coupler = None
        self._fluidServer = None
        self._solidServer = None

        self.generator = None
        
        return


    def _configure(self):
        Solver._configure(self)
        self.generator = self.inventory.generator
        return


# version
__id__ = "$Id: PressurePulse.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

# End of file
