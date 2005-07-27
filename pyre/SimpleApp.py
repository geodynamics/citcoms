#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from mpi.Application import Application
import journal


class SimpleApp(Application):


    def __init__(self, name="citcom"):
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



# main
if __name__ == "__main__":

    app = SimpleApp("citcoms")
    app.run()




# version
__id__ = "$Id: SimpleApp.py,v 1.12 2005/07/27 01:58:31 leif Exp $"

# End of file
