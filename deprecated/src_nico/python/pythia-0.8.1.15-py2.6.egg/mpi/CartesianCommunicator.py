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


raise NotImplementedError() # 2007-06-11 lcs


from Communicator import Communicator


class CartesianCommunicator(Communicator):


    def coordinates(self):
        return self._coordinates


    def shift(self, axis, direction):
        import _mpi
        return _mpi.cartesianShift(self._handle, axis, direction)


    def __init__(self, handle, axes, periods, reorder):
        import _mpi
        # create the handle
        cartesian = _mpi.communicatorCreateCartesian(handle, reorder, axes, periods)

        # create the python wrapper
        Communicator.__init__(self, cartesian)

        # cache attributes
        self._axes = axes
        self._periods = periods
        self._coordinates = _mpi.communicatorCartesianCoordinates(
            self._handle, self.rank, len(axes))

        return


# version
__id__ = "$Id: CartesianCommunicator.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
