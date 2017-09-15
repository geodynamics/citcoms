#! usr/bin/env python
#
#
#
# need corresponding coupler controller solver
#
#
#
#
#
#

from BaseApplication import BaseApplication
import journal

class MultiCoupledApp(BaseApplication):


    def __init__(self, name="CitcomS"):
        BaseApplication.__init__(self, name)

        self.solver = None
        self.solverCommunicator = None

        # list of communicators used to pass imformation between solvers
        self.myPlus = []
        self.remotePlus = []

        # containing solver need to do more communication
        self.myPlus2 = []
        self.remotePlus2 = []


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
        s3 = self.inventory.esolver2.inventory.mesher.inventory
        nproc3 = s3.nproc_surf * s3.nprocx * s3.nprocy * s3.nprocz

        return nproc1 + nproc2 + nproc3


    def initialize(self):

        layout = self.inventory.layout
        layout.initialize(self)

        self.findLayout(layout)

        self.controller.initialize(self)

        return


    def findLayout(self, layout):

        if layout.ccomm:
            # This process belongs to the containing solver
            self.controller = self.inventory.ccontroller
            self.solver = self.inventory.csolver
            self.coupler = self.inventory.ccoupler
            self.solverCommunicator = layout.ccomm
            self.myPlus = layout.ccommPlus1
            self.remotePlus = layout.ecommPlus1
            self.myPlus2 = layout.ccommPlus2
            self.remotePlus2 = layout.ecommPlus2


        elif layout.ecomm1:
            # This process belongs to the embedded solver1
            self.controller = self.inventory.econtroller1
            self.solver = self.inventory.esolver1
            self.coupler = self.inventory.ecoupler1
            self.solverCommunicator = layout.ecomm1
            self.myPlus = layout.ecommPlus1
            self.remotePlus = layout.ccommPlus1


        elif layout.ecomm2:
            # This process belongs to the embedded solver2
            self.controller = self.inventory.econtroller2
            self.solver = self.inventory.esolver2
            self.coupler = self.inventory.ecoupler2
            self.solverCommunicator = layout.ecomm2
            self.myPlus = layout.ecommPlus2
            self.remotePlus = layout.ccommPlus2

        else:
            # This process doesn't belong to any solver
            import journal
            journal.warning(self.name).log("node '%d' is an orphan"
                                           % layout.rank)

        self.comm = layout.comm
        self.rank = layout.rank
        self.nodes = layout.nodes

        return


    def reportConfiguration(self):

        rank = self.comm.rank

        if rank != 0:
            return

        self._info.line("configuration:")

        self._info.line("  facilities:")
        self._info.line("    launcher: %r" % self.inventory.launcher.name)

        self._info.line("    csolver: %r" % self.inventory.csolver.name)
        self._info.line("    esolver1: %r" % self.inventory.esolver1.name)
        self._info.line("    esolver2: %r" % self.inventory.esolver2.name)
        self._info.line("    ccontroller: %r" % self.inventory.ccontroller.name)
        self._info.line("    econtroller1: %r" % self.inventory.econtroller1.name)
        self._info.line("    econtroller2: %r" % self.inventory.econtroller2.name)
        self._info.line("    ccoupler: %r" % self.inventory.ccoupler.name)
        self._info.line("    ecoupler1: %r" % self.inventory.ecoupler1.name)
        self._info.line("    ecoupler2: %r" % self.inventory.ecoupler2.name)
        self._info.line("    layout: %r" % self.inventory.layout.name)

        return


    class Inventory(BaseApplication.Inventory):

        import pyre.inventory

        import Controller
        import Solver
        import Coupler
        import MultiLayout

        ccontroller = pyre.inventory.facility(name="ccontroller",
                                              factory=Controller.controller,
                                              args=("ccontroller","ccontroller"))
        econtroller1 = pyre.inventory.facility(name="econtroller1",
                                               factory=Controller.controller,
                                               args=("econtroller1","econtroller1"))
        econtroller2 = pyre.inventory.facility(name="econtroller2",
                                               factory=Controller.controller,
                                               args=("econtroller2","econtroller2"))

        ccoupler = pyre.inventory.facility("ccoupler",
                                           factory=Coupler.multicontainingcoupler,
                                           args=("ccoupler","ccoupler"))
        ecoupler1 = pyre.inventory.facility("ecoupler1",
                                            factory=Coupler.multiembeddedcoupler,
                                            args=("ecoupler1","ecoupler1"))
        ecoupler2 = pyre.inventory.facility("ecoupler2",
                                            factory=Coupler.multiembeddedcoupler,
                                            args=("ecoupler2","ecoupler2"))

        csolver = pyre.inventory.facility("csolver",
                                          factory=Solver.multicoupledRegionalSolver,
                                          args=("csolver", "csolver"))
        esolver1 = pyre.inventory.facility("esolver1",
                                           factory=Solver.multicoupledRegionalSolver,
                                           args=("esolver1", "esolver1"))
        esolver2 = pyre.inventory.facility("esolver2",
                                           factory=Solver.multicoupledRegionalSolver,
                                           args=("esolver2", "esolver2"))

        layout = pyre.inventory.facility("layout", factory=MultiLayout.MultiLayout,
                                         args=("layout", "layout"))

        steps = pyre.inventory.int("steps", default=1)


# version
__id__ = "$Id: MultiCoupledApp.py 7714 2007-07-19 18:51:12Z tan2 $"

# End of file
