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


from SynchronizedExchanger import SynchronizedExchanger


class ICEExchanger(SynchronizedExchanger):


    def root(self, path=None):
        if path is not None:
            self._root = path

        import elc
        elc.xdmfOpen(self._xdmf, self._root, "w")
        elc.xdmfClose(self._xdmf)

        return self._root


    def sendBoundary(self):
        self._sendNodeCount()
        self._sendFacetCount()
        self._sendCoordinates()
        self._sendConnectivity()
        return
        

    def sendVelocities(self):
        self._sendVelocities()
        return
        

    def sendPressures(self):
        self._sendPressure()
        return
        

    def receiveBoundary(self):
        self._receiveNodeCount()
        self._receiveFacetCount()
        self._receiveCoordinates()
        self._receiveConnectivity()
        return
        

    def receiveVelocities(self):
        self._receiveVelocities()
        return
        

    def receivePressures(self):
        self._receivePressure()
        return
        
    
    def __init__(self, name=None):
        if name is None:
            name = "ice"
            
        import journal
        journal.firewall("elc.ICE").log("NYI: this exchanger may not function properly")
        
        SynchronizedExchanger.__init__(self, name)

        self._root = None

        import elc
        self._xdmf = elc.xdmf()

        return


    def _sendNodeCount(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/nodes", "rw")
        elc.putNodeCountICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _receiveNodeCount(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/nodes", "r")
        elc.getNodeCountICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _sendFacetCount(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/facets", "rw")
        elc.putFacetCountICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _receiveFacetCount(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/facets", "r")
        elc.getFacetCountICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _sendCoordinates(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/coordinates", "rw")
        elc.putCoordinatesICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _receiveCoordinates(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/coordinates", "r")
        elc.getCoordinatesICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _sendConnectivity(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/connectivity", "rw")
        elc.putConnectivityICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _receiveConnectivity(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/connectivity", "r")
        elc.getConnectivityICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _sendVelocities(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/velocities", "rw")
        elc.putVelocitiesICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _receiveVelocities(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/velocities", "r")
        elc.getVelocitiesICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _sendPressure(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/pressure", "rw")
        elc.putPressureICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


    def _receivePressure(self):
        import elc
        elc.xdmfOpen(self._xdmf, self._root + "/pressure", "r")
        elc.getPressureICE(self._boundary, self._xdmf)
        elc.xdmfClose(self._xdmf)
        return


# version
__id__ = "$Id: ICEExchanger.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

#  End of file 
