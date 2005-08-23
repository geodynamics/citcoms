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

from pyre.components.Component import Component


class Exchanger(Component):


    def __init__(self, name, facility):
        Component.__init__(self, name, facility)

        self.module = None
        self.mesh = None
        self.all_variables = None
        self.communicator = None
        self.srcComm = []
        self.sinkComm = None
        self.numSrc = 0

        self.sink = {}
        self.source = {}

        self.catchup = True
        self.done = False
        self.coupled_steps = 1
        return


    def initialize(self, solver):
        self.selectModule()
        self.communicator = solver.communicator
        self.srcComm = solver.myPlus
        self.numSrc = len(self.srcComm)

        # only one of remotePlus is sinkComm
        self.sinkComm = solver.remotePlus[self.communicator.rank]
        return


    def launch(self, solver):
        self.createMesh()
        self.createSourceSink()
        self.createBC()

        if self.inventory.two_way_communication:
            self.createII()
        return


    def selectModule(self):
        import CitcomS.Exchanger
        self.module = CitcomS.Exchanger
        return


    def modifyT(self, bbox):
        self.module.modifyT(bbox, self.all_variables)
        return


    def preVSolverRun(self):
        # do nothing, overridden by FGE
        return


    def postVSolverRun(self):
        # do nothing, overridden by CGE
        return


    def endTimestep(self, steps, done):
        KEEP_WAITING_SIGNAL = 0
        NEW_STEP_SIGNAL = 1
        END_SIMULATION_SIGNAL = 2

        if done:
            sent = END_SIMULATION_SIGNAL
        elif self.catchup:
            sent = NEW_STEP_SIGNAL
        else:
            sent = KEEP_WAITING_SIGNAL

        while 1:
            signal = self.exchangeSignal(sent)

            if done or (signal == END_SIMULATION_SIGNAL):
                done = True
                break
            elif signal == KEEP_WAITING_SIGNAL:
                pass
            elif signal == NEW_STEP_SIGNAL:
                if self.catchup:
                    #print self.name, 'exchanging timestep =', steps
                    self.coupled_steps = self.exchangeSignal(steps)
                    #print self.name, 'exchanged timestep =', self.coupled_steps
                break
            else:
                raise ValueError, \
                      "Unexpected signal value, singnal = %d" % signal

        return done



    class Inventory(Component.Inventory):

        import pyre.inventory as prop



        two_way_communication = prop.bool("two_way_communication", default=True)

        # if dimensional is True, quantities exchanged are dimensional
        dimensional = prop.bool("dimensional", default=True)
        # if transformational is True, quantities exchanged are in standard coordiate system
        transformational = prop.bool("transformational", default=True)




# version
__id__ = "$Id: Exchanger.py,v 1.23 2005/06/10 02:23:22 leif Exp $"

# End of file
