#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def hello():
    from mpi import _mpi as rawmpi

    rank = rawmpi.communicatorRank(rawmpi.world)
    size = rawmpi.communicatorSize(rawmpi.world)

    print "Hello: this is %03d/%03d" % (rank, size)
    
    return


# main

if __name__ == "__main__":
    import journal
    journal.debug("mpi").activate()
    journal.debug("mpi.init").activate()
    journal.debug("mpi.fini").activate()

    hello()


# version
__id__ = "$Id: raw.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
