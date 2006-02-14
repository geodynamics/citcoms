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
    world = mpi.world()
    
    printGroup("world", world, world.group())
    testExclude("world", world)
    testInclude("world", world)

    return


def testExclude(name, communicator):
    excluded = [1]
    new = communicator.exclude(excluded)

    if new:
        printGroup("exclude", new, new.group())
    else:
        printGroup("%s - excluded" % name, communicator, communicator.group())

    return


def testInclude(name, communicator):
    included = [1]
    new = communicator.include(included)

    if new:
        printGroup("include", new, new.group())
    else:
        printGroup("%s - excluded" % name, communicator, communicator.group())

    return


def printGroup(name, communicator, group):
    print tag(name, communicator), "group: %03d/%03d" % (group.rank, group.size)
    return


def tag(name, communicator):
    return "[%s %03d/%03d]:" % (name, communicator.rank, communicator.size)


# main

if __name__ == "__main__":
    test()


# version
__id__ = "$Id: groups.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file
