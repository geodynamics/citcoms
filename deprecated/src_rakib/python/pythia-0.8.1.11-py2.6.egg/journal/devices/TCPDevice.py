#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Device import Device


class TCPDevice(Device):


    def record(self, entry):

        if self._connection is None:
            return
        
        import journal
        request = journal.request(command="record", args=[self.renderer.render(entry)])

        try:
            self._marshaller.send(request, self._connection)
            result = self._marshaller.receive(self._connection)
        except self._marshaller.RequestError:
            return

        return


    def __init__(self, key, port, host=''):
        import socket
        from NetRenderer import NetRenderer

        Device.__init__(self, NetRenderer())

        self.host = host
        self.port = port

        import journal
        self._marshaller = journal.pickler()
        self._marshaller.key = key

        import pyre.ipc
        self._connection = pyre.ipc.connection('tcp')

        self._connection.connect((self.host, self.port))

        return


# version
__id__ = "$Id: TCPDevice.py,v 1.3 2005/03/14 22:59:18 aivazis Exp $"

# End of file
