#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
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
        import ExchangerLib
        if layout.coarse:
            self.exchanger = self.inventory.coarse
            self.communicator = layout.coarse
            self.all_variables = ExchangerLib.CoarsereturnE()
        elif layout.fine:
            self.exchanger = self.inventory.fine
            self.communicator = layout.fine
            self.all_variables = ExchangerLib.FinereturnE()
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

        import CitcomS.Coupler as Coupler
        import CitcomS.Controller as Controller
        import CitcomS.Layout as Layout


        controller = pyre.inventory.facility("controller", default=Controller.controller())
        layout = pyre.inventory.facility("layout", default=Layout.layout())

        coarse = pyre.inventory.facility("coarse", factory=Coupler.containingcoupler,
                                         args=("coarse", "coarse"))
        fine = pyre.inventory.facility("fine", factory=Coupler.embeddedcoupler,
                                         args=("fine","fine"))




# main

if __name__ == "__main__":

    import mpi

    # testing Exchangermodule.so
    import ExchangerLib
    if not mpi.world().rank:
        print ExchangerLib.copyright()
        print dir(ExchangerLib)

    import journal
    #journal.debug("Array2D").activate()
    journal.debug("Exchanger").activate()
    journal.info("  X").activate()
    journal.info("  proc").activate()
    journal.info("  bid").activate()

    app = TestExchanger("test")
    app.main()



# version
__id__ = "$Id$"

# End of file
