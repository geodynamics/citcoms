#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class Communicator:


    def handle(self):
        return self._handle


    def barrier(self):
        from mpi import MPI_Barrier
        MPI_Barrier(self._handle)


    # communicator factories
    def cartesian(self, axes, periods, reorder=1):
        from CartesianCommunicator import CartesianCommunicator
        return CartesianCommunicator(self._handle, axes, periods, reorder)


     # communicator group interface
    def group(self):
        from mpi import MPI_Comm_group
        from CommunicatorGroup import CommunicatorGroup
        grpHandle = MPI_Comm_group(self._handle)
        return CommunicatorGroup(grpHandle)


    def include(self, included):
        from mpi import MPI_Comm_create, MPI_COMM_NULL
        grp = self.group().include(included)
        handle = MPI_Comm_create(self._handle, grp.handle())
        if handle is MPI_COMM_NULL:
            return None
        return Communicator(handle)


    def exclude(self, excluded):
        from mpi import MPI_Comm_create, MPI_COMM_NULL
        grp = self.group().exclude(excluded)
        handle = MPI_Comm_create(self._handle, grp.handle())
        if handle is MPI_COMM_NULL:
            return None
        return Communicator(handle)


    # ports
    def port(self, peer, tag):
        from Port import Port
        return Port(self, peer, tag)

    def __init__(self, handle):

        self._handle = handle
        
        from mpi import MPI_Comm_rank, MPI_Comm_size
        self.rank = MPI_Comm_rank(self._handle)
        self.size = MPI_Comm_size(self._handle)
        
        return


#
# Construct the pre-defined world communicator
#

def world():
    global _mpi_world
    if not _mpi_world:
        from mpi import MPI_COMM_WORLD
        _mpi_world = Communicator(MPI_COMM_WORLD)
    return _mpi_world


# the singleton

_mpi_world = None


# version
__id__ = "$Id: Communicator.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
