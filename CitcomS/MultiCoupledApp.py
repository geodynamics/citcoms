#! usr/bin/env python
#
#
#
#
#
#
#
#
#
#

from BasApplication import Base Application
import journal

class CoupledA(BaseApplication):


    def __init__(self, name="MultiCoupledCitcomS"):
        BaseApplication.__init__(self, name)
        '''
        self.solver = None
        self.solverCommunicator = None 
        self.myPlus = [] 
        self.remotePlus = [] 

        self.comm = None 
        self.rank = 0
        self.nodes = 0
        '''

        
        return

    def getNodes(self):
        # csolver requires nproc1 CPUs to run

        # esolver1 requires nproc2 CPUs to run

        # esolver2 requires nproc3 CPUs to run

        return nproc1 + nproc2 + nproc3

    def initialize(self):
        '''
        layout = self.inventory.layout
        layout.initialize(self)

        seelf.findLayout(layout)

        self.comtroller.initialize(self)
        '''

        return

    def findLayout(self, layout):

        if layout.ccomm:
        #This process belongs to the containing solver

        if layout.ecomm1:
        #This process belongs to the embedded solver1
        
        if layout.ecomm2:
        #This process belongs to the embedded solver2

        self.comm = layout.comm
        self.rank = layout.rank
        self.nodes = layout.nodes

        return

    
    def report Configuration(self):

        return

    class Invertory(BaseApplication.Inventory):

        import pyre.inventory

        import Controller
        import Solver
        import Coupler
        import Layout

        ## pyre.inventory stuff

# End of file
        
