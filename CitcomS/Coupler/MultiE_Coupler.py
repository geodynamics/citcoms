#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


#
#embedded coupler used in multi coupled application
#
#

from EmbeddedCoupler import EmbeddedCoupler

class MultiE_Coupler(EmbeddedCoupler):

    def endTimestep(self, steps, done):
        # exchange predefined signal btwn couplers
        # the signal is used to sync the timesteps
        KEEP_WAITING_SIGNAL = 0
        NEW_STEP_SIGNAL = 1
        END_SIMULATION_SIGNAL = 2
        BIG_NEW_STEP_SIGNAL = 3

        if done:
            sent = END_SIMULATION_SIGNAL
        elif self.synchronized:
            sent = NEW_STEP_SIGNAL
        else:
            sent = KEEP_WAITING_SIGNAL

        while 1:

            #print "*****" , self.name
            # send signal
            recv = self.exchangeSignal(sent)
            #print "ecplr send %d" % sent

            # receive instruction
            recv = self.exchangeSignal(sent)
            #print "ecplr receive %d" % recv
            #print "*****"

            # determine what to do
            if done or (recv == END_SIMULATION_SIGNAL):
                done = True
                break
            elif recv == NEW_STEP_SIGNAL:
                # keep going until synchronized
                if self.synchronized:
                    pass
                else:
                    break
            elif recv == BIG_NEW_STEP_SIGNAL:
                assert self.synchronized, \
                       "embedded coupler not synchronized on a big step"
                #print self.name, 'exchanging timestep =', steps
                self.coupled_steps = self.exchangeSignal(steps)
                #print self.name, 'exchanged timestep =', self.coupled_steps
                break
            else:
                raise ValueError, \
                      "Unexpected signal value, singnal = %d" % recv

        return done

