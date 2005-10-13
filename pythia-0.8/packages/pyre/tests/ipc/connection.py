#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def main():


    from pyre.applications.Script import Script


    class ConnectionApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            port = pyre.inventory.int('port', default=50000)
            host = pyre.inventory.str('host', default='localhost')

            protocol = pyre.inventory.str("protocol", default="udp")


        def main(self, *args, **kwds):
            import journal
            journal.info("pyre.ipc.connection").activate()

            host = self.inventory.host
            port = self.inventory.port
            protocol = self.inventory.protocol
            
            import pyre.ipc
            connection = pyre.ipc.connection(protocol)
            connection.connect((self.inventory.host, self.inventory.port))

            outstream = connection.makefile("wb")

            import pickle
            hello = pickle.dumps("Hello world!")
            outstream.write(hello)
            outstream.flush()
            
            return


        def __init__(self):
            Script.__init__(self, 'connection')
            return


    app = ConnectionApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: connection.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
