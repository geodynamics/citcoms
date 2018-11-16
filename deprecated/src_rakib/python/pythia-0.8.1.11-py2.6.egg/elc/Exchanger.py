#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from pyre.components.Component import Component


class Exchanger(Component):


    def initialize(self, communicator=None):
        if communicator is None:
            import mpi
            communicator = mpi.world()
            
        self.communicator = communicator
        self.rank = self.communicator.rank

        return


    def setMesh(self, mesh):
        # register the triangulation
        self.boundary.setMesh(mesh)
        return
    

    def exchange(self):
        self.exchangeBoundary()
        # self.exchangeFields()

        return


    def exchangeBoundary(self):
        rank = self.rank
        self._info.log("rank %d: exchanging boundary %d -> %d" % (rank, self.source, self.sink))
        if rank == self.source:
            self._info.log("sending boundary to node %d" % self.sink)
            self.sendBoundary()

        elif rank == self.sink:
            self._info.log("receiving boundary from node %d" % self.source)
            self.receiveBoundary()
        
        self._info.log("done exchanging boundary %d -> %d" % (self.source, self.sink))

        return


    def exchangeFields(self):
        self.exchangeVelocities()
        self.exchangePressure()
        return
    

    def exchangeVelocities(self):
        rank = self.rank

        self._info.log("exchanging velocities %d -> %d" % (self.source, self.sink))
        if rank == self.source:
            self._info.log("sending velocities to node %d" % self.sink)
            self.sendVelocities()

        elif rank == self.sink:
            self._info.log("receiving velocities from node %d" % self.source)
            self.receiveVelocities()
        
        self._info.log("done exchanging velocities %d -> %d" % (self.source, self.sink))

        return


    def exchangePressure(self):
        rank = self.rank

        self._info.log("exchanging pressure %d -> %d" % (self.source, self.sink))
        if rank == self.source:
            self._info.log("receiving pressures from node %d" % self.sink)
            self.receivePressures()

        elif rank == self.sink:
            self._info.log("sending pressures to node %d" % self.source)
            self.sendPressures()
        
        self._info.log("done exchanging pressure %d -> %d" % (self.source, self.sink))

        return


    def servers(self, source, sink):
        self.sink = sink
        self.source = source
        return


    def __init__(self, name):
        Component.__init__(self, name, facility="exchanger")
        
        self.sink = None
        self.source = None

        self.mesh = None
        self.rank = None
        self.communicator = None

        from Boundary import Boundary
        self.boundary = Boundary()

        import journal
        self._info = journal.debug("exchanger")
        return
        

# version
__id__ = "$Id: Exchanger.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

#  End of file 
