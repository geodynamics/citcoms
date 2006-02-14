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


    from mpi.Application import Application


    class ExchangeApp(Application):


        class Inventory(Application.Inventory):

            import elc
            import pyre.inventory

            modeller = pyre.inventory.facility('modeller', default="hollow-cube")
            coupler = pyre.inventory.facility('coupler', factory=elc.mpiExchanger)


        def main(self, *args, **kwds):
            import mpi
            world = mpi.world()
            rank = world.rank

            self.coupler.servers(0, 1)
            self.coupler.initialize(communicator=world)

            if rank == 0:
                self.onSource()
            elif rank == 1:
                self.onSink()
            else:
                import journal
                journal.firewall("exchange").log("too many processors....")

            return


        def onSource(self):
            import journal
            journal.debug("elc.exchange").activate()

            model = self.modeller.model()

            import acis
            mesher = acis.surfaceMesher()
    
            properties = mesher.inventory
            properties.gridAspectRatio = 1.0
            properties.maximumEdgeLength = 0.01
    
            mesh, bbox = mesher.facet(model)

            self.coupler.setMesh(mesh)
            self.coupler.exchangeBoundary()

            return


        def onSink(self):
            import journal
            journal.debug("elc.exchange").activate()

            self.coupler.exchangeBoundary()

            return


        def __init__(self):
            Application.__init__(self, 'exchange')
            self.coupler = None
            self.modeller = None

            return


        def _defaults(self):
            Application._defaults(self)
            self.inventory.launcher.inventory.nodes = 2
            return


        def _configure(self):
            Application._configure(self)
            self.coupler = self.inventory.coupler
            self.modeller = self.inventory.modeller
            return

    app = ExchangeApp()
    return app.run()


# main
if __name__ == '__main__':
    # journal
    
    # invoke the application shell
    main()


# version
__id__ = "$Id: exchange.py,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $"

# End of file 
