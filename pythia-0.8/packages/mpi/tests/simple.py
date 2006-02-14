#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application


class SimpleApp(Application):


    def main(self):
        import mpi
        import socket

        hostname = socket.gethostname()
        world = mpi.world()
        print "[%03d/%03d] Hello world from '%s'!" % (world.rank, world.size, hostname)

        return


    def __init__(self):
        Application.__init__(self, "simple")
        return


# main

if __name__ == "__main__":
    import journal
    journal.debug("simple").activate()
    journal.debug("launcher").activate()
    
    app = SimpleApp()
    app.run()

# version
__id__ = "$Id: simple.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file 
