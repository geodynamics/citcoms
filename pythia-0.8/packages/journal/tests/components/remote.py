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


    from pyre.applications.ClientServer import ClientServer


    class RemoteApp(ClientServer):


        class Inventory(ClientServer.Inventory):

            import pyre.inventory

            # the key used by the remote server
            key = pyre.inventory.str("key", default="deadbeef")

            # host where the journal server is located
            host = pyre.inventory.str("host", default="localhost")

            # port used by the remote server
            port = pyre.inventory.int("port", default=50000)
            
            # delay before client is spawned
            delay = pyre.inventory.int("delay", default=5)
            
            # [client, server, both]
            mode = pyre.inventory.str(
                "mode", default="both",
                validator=pyre.inventory.choice(["server", "client", "both"]))
            

        def onServer(self):
            name = 'journal'

            # turn on my channels so we can watch what happens
            import journal
            journal.info(name).activate()
            journal.debug(name).activate()

            # build the registry settings
            registry = self.createRegistry()
            registry.name = 'root'
            node = registry.getNode(name)

            node.setProperty('port', self.port, self.inventory.getTraitDescriptor('port').locator)

            marshaller = node.getNode('marshaller')
            marshaller.setProperty('key', self.key, None)

            # instantiate the services
            service = journal.service(name)

            # configure it
            self.configureComponent(service, registry)

            # initialize it
            service.init()

            # enter the indefinite loop waiting for requests
            service.serve()

            return


        def onClient(self):
            # initialize the journal device
            import journal
            journal.remote(self.key, self.port, self.host)

            for idx in range(5):
                info = journal.debug("test-%02d" % idx).activate()
                info.log("test %02d: this a sample message" % idx)

            return


        def __init__(self):
            ClientServer.__init__(self, 'remote')
            self.key = ''
            self.host = ''
            self.port = 0
            self.mode = ''
            self.delay = ''
            return


        def _configure(self):
            ClientServer._configure(self)
            self.key = self.inventory.key
            self.host = self.inventory.host
            self.port = self.inventory.port
            self.mode = self.inventory.mode
            self.delay = self.inventory.delay
            return


        def _init(self):
            ClientServer._init(self)
            return


    app = RemoteApp()
    return app.run()


# main
if __name__ == '__main__':
    import journal
    journal.info("remote").activate()
    journal.debug("remote").activate()
    
    # invoke the application shell
    main()


# version
__id__ = "$Id: remote.py,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $"

# End of file 
