#! usr/bin/env python
#
#
#
#need to work on myPlus and remotePlus
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
        
        self.solver = None
        self.solverCommunicator = None 

        #list of communicators used to pass imformation between solvers
        self.myPlus = [] 
        self.remotePlus = [] 

        self.comm = None 
        self.rank = 0
        self.nodes = 0
        

        
        return

    def getNodes(self):
        # csolver requires nproc1 CPUs to run
        s1 = self.inventory.csolver.inventory.mesher.inventory
        nproc1 = s1.nproc_surf * s1.nprocx * s1.nprocy * s1.nprocz

        # esolver1 requires nproc2 CPUs to run
        s2 = self.inventory.esolver1.inventory.mesher.inventory
        nproc2 = s2.nproc_surf * s2.nprocx * s2.nprocy * s2.nprocz

        # esolver2 requires nproc3 CPUs to run
        s3 = self.inventory.esolver2.inventory.mecher.inventory
        nproc3 = s3.nproc_surf * s3.nprox * s3.nprocy * s3.nprocz
        
        return nproc1 + nproc2 + nproc3

    def initialize(self):
        
        layout = self.inventory.layout
        layout.initialize(self)

        seelf.findLayout(layout)

        self.comtroller.initialize(self)
        

        return

    def findLayout(self, layout):

        if layout.ccomm:
            # This process belongs to the containing solver
            self.controller = self.inventory.ccontroller
            self.solver = self.inventory.csolver
            self.coupler = self.inventory.ccoupler
            self.solverCommunicator = layout.ccomm
            #self.myPlus = layout.ccommPlus
            #self.remotePlus = layout.ecommPlus
  
        elif layout.ecomm1:
            # This process belongs to the embedded solver
            self.controller = self.inventory.econtroller1
            self.solver = self.inventory.esolver1
            self.coupler = self.inventory.ecoupler1
            self.solverCommunicator = layout.ecomm1
            #self.myPlus = layout.ecommPlus
            #self.remotePlus = layout.ccommPlus
 

        elif layout.ecomm2:
            # This process belongs to the embedded solver
            self.controller = self.inventory.econtroller2
            self.solver = self.inventory.esolver2
            self.coupler = self.inventory.ecoupler2
            self.solverCommunicator = layout.ecomm2
            #self.myPlus = layout.ecommPlus
            #self.remotePlus = layout.ccommPlus
     
        else:
            # This process doesn't belong to any solver
            import journal
            journal.warning(self.name).log("node '%d' is an orphan"
                                           % layout.rank)

        self.comm = layout.comm
        self.rank = layout.rank
        self.nodes = layout.nodes

        return

    
    def report Configuration(self):

        rank = self.comm.rank

        if rank != 0:
            return

        self._info.line("configuration:")

        self._info.line("  facilities:")
        self._info.line("    launcher: %r" % self.inventory.launcher.name)

        self._info.line("    csolver: %r" % self.inventory.csolver.name)
        self._info.line("    esolver: %r" % self.inventory.esolver.name)
        self._info.line("    ccontroller: %r" % self.inventory.ccontroller.name)
        self._info.line("    econtroller: %r" % self.inventory.econtroller.name)
        self._info.line("    ccoupler: %r" % self.inventory.ccoupler.name)
        self._info.line("    ecoupler: %r" % self.inventory.ecoupler.name)
        self._info.line("    layout: %r" % self.inventory.layout.name)
    
        return

    class Invertory(BaseApplication.Inventory):

        import pyre.inventory

        import Controller
        import Solver
        import Coupler
        import Layout
        ccontroller = pyre.inventory.facility(name="ccontroller",
                                              factory=Controller.controller,
                                              args=("ccontroller","ccontroller"))
        econtroller1 = pyre.inventory.facility(name="econtroller1",
                                              factory=Controller.controller,
                                              args=("econtroller","econtroller"))
        econtroller2 = pyre.inventory.facility(name="econtroller2",
                                              factory=Controller.controller,
                                              args=("econtroller","econtroller"))


        ccoupler = pyre.inventory.facility("ccoupler",
                                           factory=Coupler.containingcoupler,
                                           args=("ccoupler","ccoupler"))
        ecoupler1 = pyre.inventory.facility("ecoupler1",
                                           factory=Coupler.embeddedcoupler,
                                           args=("ecoupler","ecoupler"))
        ecoupler2 = pyre.inventory.facility("ecoupler2",
                                           factory=Coupler.embeddedcoupler,
                                           args=("ecoupler","ecoupler"))

        csolver = pyre.inventory.facility("csolver",
                                          factory=Solver.coupledFullSolver,
                                          args=("csolver", "csolver"))
        esolver1 = pyre.inventory.facility("esolver1",
                                       factory=Solver.coupledRegionalSolver,
                                       args=("esolver", "esolver"))
        esolver2 = pyre.inventory.facility("esolver2",
                                       factory=Solver.coupledRegionalSolver,
                                       args=("esolver", "esolver"))

        layout = pyre.inventory.facility("layout", factory=MultiLayout.MultiLayout,
                                         args=("layout", "layout"))

        steps = pyre.inventory.int("steps", default=1)


        
# End of file
        
