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

from Exchanger import Exchanger


class SynchronizedExchanger(Exchanger):


    def exchangeBoundary(self, communicator=None):
        if communicator is None:
            import mpi
            communicator = mpi.world()

        rank = communicator.rank

        self._info.log("exchanging boundary %d -> %d" % (self.source, self.sink))
        if rank == self.source:
            self._info.log("sending boundary to node %d" % self.sink)
            self.sendBoundary()
            self._info.log("at barrier: done writing")
            communicator.barrier()

        elif rank == self.sink:
            self._info.log("at barrier: waiting to read")
            communicator.barrier()
            self._info.log("receiving boundary from node %d" % self.source)
            self.receiveBoundary()
        else:
            # other processors can skip this exchange
            self._info.log("rank %d: innocent bystander" % rank)
            import mpi
            if communicator == mpi.world():
                communicator.barrier()
        
        self._info.log("done exchanging boundary %d -> %d" % (self.source, self.sink))

        return


    def exchangeVelocities(self, communicator=None):
        import pyre

        if communicator is None:
            import mpi
            communicator = mpi.world()

        rank = communicator.rank

        self._info.log("exchanging velocities %d -> %d" % (self.source, self.sink))
        if rank == self.source:
            self._info.log("sending velocities to node %d" % self.sink)
            self.sendVelocities()
            self._info.log("at barrier: done writing")
            communicator.barrier()

        elif rank == self.sink:
            self._info.log("at barrier: waiting to read")
            communicator.barrier()
            self._info.log("receiving velocities from node %d" % self.source)
            self.receiveVelocities()

        else:
            # other processors can skip this exchange
            self._info.log("rank %d: innocent bystander" % rank)
            import mpi
            if communicator == mpi.world():
                communicator.barrier()
        
        self._info.log("done exchanging velocities %d -> %d" % (self.source, self.sink))

        return


    def exchangePressure(self, communicator=None):

        if communicator is None:
            import mpi
            world = mpi.world()

        rank = communicator.rank

        if not communicator:
            communicator = world

        self._info.log("exchanging pressure %d -> %d" % (self.source, self.sink))
        if rank == self.source:
            self._info.log("at barrier: waiting to read")
            communicator.barrier()
            self._info.log("receiving pressures from node %d" % self.sink)
            self.receivePressures()

        elif rank == self.sink:
            self._info.log("sending pressures to node %d" % self.source)
            self.sendPressures()
            self._info.log("at barrier: done writing")
            communicator.barrier()
        else:
            # other processors can skip this exchange
            self._info.log("rank %d: innocent bystander" % rank)
            import mpi
            if communicator == mpi.world():
                communicator.barrier()
        
        self._info.log("done exchanging pressure %d -> %d" % (self.source, self.sink))

        return


    def __init__(self, name):
        Exchanger.__init__(self, name)
        return
        

# version
__id__ = "$Id: SynchronizedExchanger.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

#  End of file 
