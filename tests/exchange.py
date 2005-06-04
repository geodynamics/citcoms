#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from mpi.Application import Application
import journal


class TestExchanger(Application):

    def __init__(self, name="citcom"):
        Application.__init__(self, name)
        self.exchanger = None
        return


    def run(self):
        layout = self.inventory.layout
        layout.initialize(self)

        self.findLayout(layout)

        exchanger = self.exchanger
        if exchanger:
            self.test(exchanger)

        return


    def test(self, exchanger):
        # testing exchanger creation
        exchanger.selectModule()
        exchanger.createExchanger(self)
        #print exchanger.name, exchanger.exchanger

        # testing boundary creation and exchange
        exchanger.findBoundary()
        print exchanger.name, ": boundary found"

        # testing applyBoundaryConditions
        exchanger.applyBoundaryConditions()
        print exchanger.name, ": applyBoundaryConditions worked"

        # testing initTemperature
        #exchanger.initTemperature()
        #print exchanger.name, ": temperature transferred"

        # select dt
        try:
            # success if exchanger is a FGE
            exchanger.fge_t
            dt = 0.4
        except:
            # exception if exchanger is a CGE
            dt = 1

        # testing dt exchanging
        print "%s - old dt = %g   exchanged dt = %g   old dt = %g" % (
              exchanger.name, dt,
              exchanger.module.exchangeTimestep(exchanger.exchanger, dt),
              dt)

        # testing exchangeSignal
        steps = 2*3 + 1
        for step in range(steps):
            time = exchanger.stableTimestep(dt)
            print "%s - step %d: %g" % (exchanger.name, step, time)
            exchanger.applyBoundaryConditions()

            if step == steps-1:
                done = True
            else:
                done = False

            done = exchanger.endTimestep(done)

            if done:
                break

        return


    def findLayout(self, layout):
        import CitcomS.Exchanger
        if layout.coarse:
            self.exchanger = self.inventory.coarse
            self.communicator = layout.coarse
            self.all_variables = CitcomS.Exchanger.CoarsereturnE()
        elif layout.fine:
            self.exchanger = self.inventory.fine
            self.communicator = layout.fine
            self.all_variables = CitcomS.Exchanger.FinereturnE()
        else:
            import journal
            journal.warning(self.name).log("node '%d' is an orphan" % layout.rank)
        self.intercomm = layout.intercomm
        self.rank = layout.rank
        self.nodes = layout.nodes
        self.leader = layout.leader
        self.remoteLeader = layout.remoteLeader

        print "%s exchanger: rank=%d  leader=%d  remoteLeader=%d" % (
              self.exchanger.name, self.rank, self.leader, self.remoteLeader)

        return




    class Inventory(Application.Inventory):

        import pyre.inventory

        import CitcomS.Components.Exchanger as Exchanger
        import CitcomS.Controller as Controller
        import CitcomS.Layout as Layout


        controller = pyre.inventory.facility("controller", default=Controller.controller())
        layout = pyre.inventory.facility("layout", default=Layout.layout())

        coarse = pyre.inventory.facility("coarse", default=Exchanger.coarsegridexchanger("coarse"))
        fine = pyre.inventory.facility("fine", default=Exchanger.finegridexchanger("fine"))




# main

if __name__ == "__main__":

    import mpi

    # testing Exchangermodule.so
    import CitcomS.Exchanger
    if not mpi.world().rank:
        print CitcomS.Exchanger.copyright()
        print dir(CitcomS.Exchanger)

    import journal
    #journal.debug("Array2D").activate()
    journal.debug("Exchanger").activate()
    journal.info("  X").activate()
    journal.info("  proc").activate()
    journal.info("  bid").activate()

    app = TestExchanger("test")
    app.main()



# version
__id__ = "$Id: exchange.py,v 1.16 2005/06/03 21:51:46 leif Exp $"

# End of file
