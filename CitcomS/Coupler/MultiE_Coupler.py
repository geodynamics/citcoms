#!/usr/bin/env python
 
 
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
            # send signal
            recv = self.exchangeSignal(sent)
            # receive instruction
            recv = self.exchangeSignal(sent)

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

