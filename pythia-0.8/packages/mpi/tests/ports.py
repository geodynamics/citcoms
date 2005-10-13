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


class PortApp(Application):


    def main(self):
        import mpi
        import socket

        hostname = socket.gethostname()
        world = mpi.world()

        rank = world.rank
        if rank == 0:
            port = world.port(peer=1, tag=17)
            port.send("Hello")
        elif rank == 1:
            port = world.port(peer=0, tag=17)
            message = port.receive()
            print "[%d/%d]: received {%s}" % (rank, world.size, message)

        return


    def __init__(self):
        Application.__init__(self, "ports")
        return


# main

if __name__ == "__main__":
    import journal
    # journal.debug("ports").activate()
    journal.debug("mpi.ports").activate()
    
    app = PortApp()
    app.run()

# version
__id__ = "$Id: ports.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file 
