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

# timers
def timingCenter():
    from TimingCenter import timingCenter
    return timingCenter()


def timer(name):
    return timingCenter().timer(name)


# attempt to load the mpi python bindings
try:
    import _mpi
except:

    def world():
        from DummyCommunicator import DummyCommunicator
        return DummyCommunicator()
    def inParallel(): return 0
    def processors(): return 1
    
else:
    from Communicator import world
    def inParallel(): return 1
    def processors(): return world().size


def copyright():
    return "pythia.mpi: Copyright (c) 1998-2005 Michael A.G. Aivazis"


# version
__version__ = "0.8"
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
