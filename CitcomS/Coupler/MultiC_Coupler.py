#!/usr/bin/env python

#
#containing coupler with more than one coupled embedded coupler
#

from ContainingCoupler import ContainingCoupler

class MultiC_coupler(ContainingCoupler):


    def __init__(self, name, facility):
        ContainingCoupler.__init__(self, name, facility)

        self.srcCommList2 = []
        self.sinkComm2 = None
        self.remoteSize2 = 0
        return


    def initialize(self, solver):
        ContainingCoupler.initialize(self, solver)

        self.srcCommList2 = solver.myPlus2

        # number of processors in the remote solver2
        self.remoteSize2 = len(self.srcCommList2)

#?        # only one of remotePlus2 is sinkComm2 
#?        self.sinkComm2 = solver.remotePlus2[self.communicator.rank]
        
        # allocate space
        self.remoteBdryList2 = range(self.remoteSize2)
        self.sourceList2 = range(self.remoteSize2)
        self.outletList2 = range(self.remoteSize2)

        ###
        return


    def createMesh(self):
        # Create BoundedMesh objects.

        ContainingCouple.createMesh(self)
        '''
        # the bounding box of the mesh on remote solver2
        self.remoteBBox2 = \
                         exchangeBoundedBox(self.globalBBox,
                                            self.communicator.handle(),
                                            self.srcCommList2[0].handle(),
                                            self.srcCommList2[0].size - 1)


        # the nodes within remoteBBox2
        self.interior2, self.myBBox2 = createInterior(self.remoteBBox2,
                                                    self.all_variables)

        # an empty boundary object,\
        # which will be filled by a remote boundary obj.
        for i in range(self.remoteSize2):
            self.remoteBdryList2[i] = createEmptyBoundary()
    
        '''
        return
    
# version

__id__="$Id:$"

# End of file


        
