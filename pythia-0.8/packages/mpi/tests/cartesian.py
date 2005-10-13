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


def test():
    import mpi

    axes = (2, 1, 1)
    periods = (0, 0, 0)

    world = mpi.world()

    if world.rank == 0:
        import journal
        journal.debug("mpi.fini").activate()
        journal.debug("mpi.cartesian").activate()
        
    cartesian = world.cartesian(axes, periods)

    wr, ws = world.rank, world.size
    cr, cs = cartesian.rank, cartesian.size
    coordinates = cartesian.coordinates()

    print "Hello: world(%03d/%03d), cartesian(%03d/%03d -> %s)" % (
        wr, ws, cr, cs, coordinates)

    return
    

# main

if __name__ == "__main__":
    test()


# version
__id__ = "$Id: cartesian.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
