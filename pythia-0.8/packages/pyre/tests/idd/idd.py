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


    class IDDApp(ClientServer):


        class Inventory(ClientServer.Inventory):

            import pyre.inventory

            # the key used by the remote server
            key = pyre.inventory.str("key", default="deadbeef")

            # host where the journal server is located
            host = pyre.inventory.str("host", default="localhost")

            # port used by the remote server
            port = pyre.inventory.int("port", default=50000)

            # the name of the client session component
            clientName = pyre.inventory.str('client-name', default='idd-session')
            

        def onServer(self):
            name = 'idd'

            # turn on my channels so we can watch what happens
            import journal
            journal.info(name).activate()
            # journal.debug(name).activate()
            journal.info('pickler').activate()
            # journal.debug('pickler').activate()

            # build the registry settings
            registry = self.createRegistry()
            registry.name = 'root'
            node = registry.getNode(name)
            node.setProperty('port', self.port,  None)

            marshaller = node.getNode('marshaller')
            marshaller.setProperty('key', self.key, None)

            # instantiate the services
            import pyre.idd
            service = pyre.idd.service(name)
            service.weaver = self.weaver

            # configure it
            self.configureComponent(service, registry)

            # initialize it
            service.init()

            # client configuration
            registry = self.createRegistry()
            serviceRegistry = registry.getNode(self.clientName)
            service.generateClientConfiguration(serviceRegistry)

            stream = file(self.clientName + '.pml', 'w')
            document = self.weaver.render(registry)
            print >> stream, "\n".join(document)
            stream.close()

            # enter the indefinite loop waiting for requests
            service.serve()

            return


        def onClient(self):
            name = self.clientName

            import journal
            journal.info(name).activate()
            # journal.debug(name).activate()
            journal.info('pickler').activate()
            # journal.debug('pickler').activate()

            # instantiate the client
            import pyre.idd
            client = pyre.idd.session(name)

            # configure it
            self.configureComponent(client)

            # initialize it
            client.init()

            # get a token
            token = client.token()
            print "got token:", token

            return


        def __init__(self, name):
            ClientServer.__init__(self, name)
            self.key = ''
            self.host = ''
            self.port = 0
            self.clientName = ''
            return


        def _configure(self):
            ClientServer._configure(self)
            self.key = self.inventory.key
            self.host = self.inventory.host
            self.port = self.inventory.port
            self.clientName = self.inventory.clientName
            return


    name = "iddapp"

    import journal
    journal.info(name).activate()
    
    app = IDDApp(name)
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: idd.py,v 1.2 2005/03/10 22:20:36 aivazis Exp $"

# End of file 
