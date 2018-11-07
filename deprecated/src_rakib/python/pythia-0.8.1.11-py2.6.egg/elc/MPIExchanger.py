#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from Exchanger import Exchanger


class MPIExchanger(Exchanger):


    def sendBoundary(self):
        import elc
        elc.sendBoundaryMPI(self.boundary.mesh.handle(), self.source, self.sink)
        return
        

    def sendVelocities(self):
        dim, order, nodes, vertices = self.boundary.mesh.statistics()
        
        length = 3*nodes
        field = self.boundary.velocity

        import elc
        elc.sendFieldMPI(self.source, self.sink, field, length)
        return
        

    def sendPressures(self):
        dim, order, nodes, vertices = self.boundary.mesh.statistics()
        
        length = nodes
        field = self.boundary.pressure

        import elc
        # source and sink are reversed for this field
        elc.sendFieldMPI(self.sink, self.source, field, length)
        return
        

    def receiveBoundary(self):
        import pyre.geometry
        mesh = pyre.geometry.mesh(3,3)
        
        import elc
        elc.receiveBoundaryMPI(mesh.handle(), self.source, self.sink)

        self.boundary.setMesh(mesh)

        return
        

    def receiveVelocities(self):
        dim, order, nodes, vertices = self.boundary.mesh.statistics()
        
        length = 3*nodes
        field = self.boundary.velocity

        import elc
        elc.receiveFieldMPI(self.source, self.sink, field, length)
        return
        

    def receivePressures(self):
        dim, order, nodes, vertices = self.boundary.mesh.statistics()
        
        length = nodes
        field = self.boundary.pressure

        import elc
        # source and sink are reversed for this field
        elc.receiveFieldMPI(self.sink, self.source, field, length)
        return
        

    def __init__(self, name=None):
        if name is None:
            name = "mpi"

        Exchanger.__init__(self, name)

        return


# version
__id__ = "$Id: MPIExchanger.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

#  End of file 
