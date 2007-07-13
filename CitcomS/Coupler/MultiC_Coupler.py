#!/usr/bin/env python

#
#containing coupler with more than one coupled embedded coupler
#

from ContainingCoupler import ContainingCoupler

class MultiC_coupler(ContainingCoupler):


    def __init__(self, name, facility):
        ContainingCoupler.__init__(self, name, facility)
        return


    def initialize(self, solver):
        ContainingCoupler.initialize(self, solver)

        ###
        return


# version

__id__=""

# End of file


        
