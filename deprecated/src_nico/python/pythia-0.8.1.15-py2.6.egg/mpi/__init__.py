#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from _mpi import *
from Application import Application


def world():
    import Communicator
    return Communicator.world()


def mpistart(argv=None, **kwds):
    """entry point for MPI applications"""

    import sys
    from pyre.applications import start, AppRunner
    from mpi import MPI_Comm_rank, MPI_COMM_WORLD

    rank = MPI_Comm_rank(MPI_COMM_WORLD)
    macros = {
        'rank': ("%04d" % rank),
        }

    kwds = kwds.get('kwds', dict())
    kwds['message'] = '_onComputeNodes'
    kwds['macros'] = macros

    try:
        start(argv,
              applicationClass = AppRunner,
              kwds = kwds)
    except:
        #MPI_Abort(MPI_COMM_WORLD, 1)
        raise
    
    return 0


# end of file
