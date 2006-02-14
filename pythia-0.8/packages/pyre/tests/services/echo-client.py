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


    class EchoApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            key = pyre.inventory.str('key', default='deadbeef')
            port = pyre.inventory.int('port', default=50000)
            host = pyre.inventory.str('host', default='localhost')


        def main(self, *args, **kwds):
            import pyre.services
            request = pyre.services.request(command='echo', args=['Hello world!'])
            self._marshaller.send(request, self._connection)
            return


        def __init__(self, name):
            Script.__init__(self, name)

            self._connection = None
            self._marshaller = None

            import journal
            journal.info("pyre.ipc.connection").activate()

            return


        def _init(self):
            Script._init(self)
            
            key = self.inventory.key
            host = self.inventory.host
            port = self.inventory.port

            import pyre.ipc
            self._connection = pyre.ipc.connection('udp')
            self._connection.connect((host, port))

            import pyre.services
            self._marshaller = pyre.services.pickler()
            self._marshaller.key = key

            return


    name = 'echoapp'   
    app = EchoApp(name)
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: echo-client.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

# End of file 
