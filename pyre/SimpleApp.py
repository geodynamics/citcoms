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


class SimpleApp(Application):


    def __init__(self, name="CitcomS"):
        Application.__init__(self, name)

        self.solver = None
        self.solverCommunicator = None
        self._info = journal.debug("application")
        return



    def main(self, *args, **kwds):
        self.initialize()
        self.reportConfiguration()
        self.launch()
        return



    def initialize(self):
        layout = self.findLayout()

        self.controller.initialize(self)
        return



    def launch(self):
        self.controller.launch(self)

        self.controller.march(steps=self.inventory.steps)
        return



    def findLayout(self):
        self.controller = self.inventory.controller
        self.solver = self.inventory.solver
        import mpi
        self.solverCommunicator = mpi.world()
        return



    def reportConfiguration(self):

        import mpi
        rank = mpi.world().rank

        if rank != 0:
            return

        self._info.line("configuration:")
        self._info.line("  properties:")
        #self._info.line("    name: %r" % self.inventory.name)
        #self._info.line("    full name: %r" % self.inventory.fullname)

        self._info.line("  facilities:")
        self._info.line("    journal: %r" % self.inventory.journal.name)
        self._info.line("    launcher: %r" % self.inventory.launcher.name)

        self._info.line("    solver: %r" % self.solver.name)
        self._info.line("    controller: %r" % self.controller.name)

        return


    def usage(self):
        name = 'citcomsregional.sh'
        if self.inventory.solver.name == 'full':
            name = 'citcomsfull.sh'
        print 'usage: %s [<property>=<value>] [<facility>.<property>=<value>] ...' % name
        self.showUsage()
        print """\
For more information about a particular component:
  --<facility>.help-properties
  --<facility>.help-components
where <facility> is the facility to which the component is bound; e.g.:
  %(name)s --launcher.help-properties""" % locals()
        return


    class Inventory(Application.Inventory):

        import pyre.inventory

        import Controller
        import Solver

        launcher = pyre.inventory.facility("launcher", default="mpich")

        controller = pyre.inventory.facility("controller", factory=Controller.controller)
        solver = pyre.inventory.facility("solver", factory=Solver.regionalSolver)

        steps = pyre.inventory.int("steps", default=1)


    def _getPrivateDepositoryLocations(self):
        list = []
        try:
            from config import makefile
            list.append(makefile['pkgsysconfdir'])
        except ImportError, KeyError:
            pass
        return list


# main
if __name__ == "__main__":

    app = SimpleApp("CitcomS")
    app.run()




# version
__id__ = "$Id$"

# End of file
