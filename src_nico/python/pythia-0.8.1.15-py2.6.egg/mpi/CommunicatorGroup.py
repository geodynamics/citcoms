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


class CommunicatorGroup:


    def handle(self):
        return self._handle


    def include(self, included):
        from mpi import MPI_Group_incl
        handle = MPI_Group_incl(self._handle, included)
        return CommunicatorGroup(handle)


    def exclude(self, excluded):
        from mpi import MPI_Group_excl
        handle = MPI_Group_excl(self._handle, excluded)
        return CommunicatorGroup(handle)


    def __init__(self, handle):

        self._handle = handle
        
        from mpi import MPI_Group_rank, MPI_Group_size
        self.rank = MPI_Group_rank(self._handle)
        self.size = MPI_Group_size(self._handle)
        
        return


# version
__id__ = "$Id: CommunicatorGroup.py,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $"

# End of file
