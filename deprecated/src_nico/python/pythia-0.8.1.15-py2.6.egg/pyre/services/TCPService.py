#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

from Service import Service


class TCPService(Service):


    def onConnectionAttempt(self, selector, monitor):
        self._debug.log("detected activity on port %d" % self.port)

        socket, address = monitor.accept()
        if not self.validateConnection(address):
            return True

        self._info.log("new connection from [%d@%s]" % (address[1], address[0]))

        # Create a closure to remember 'address'.
        def handler(selector, socket):
            return self.onRequest(selector, socket, address)
        
        selector.notifyOnReadReady(socket, handler)

        return True


    def onRequest(self, selector, socket, address):

        try:
            request = self.marshaller.receive(socket)
        except ValueError, msg:
            return self.badRequest(socket, address, msg)
        except self.marshaller.RequestError, msg:
            self.requestError(socket, address, msg)
            return False

        self._info.log("request from [%d@%s]: command=%r, args=%r" % (
            address[1], address[0], request.command, request.args))

        result = self.evaluator.evaluate(self, request.command, request.args)

        self._debug.log("got result: %s" % result)

        try:
            self.marshaller.send(result, socket)
        except self.marshaller.RequestError, msg:
            self.requestError(socket, address, msg)
            return False

        return True


    def badRequest(self, socket, address, msg):
        """Notify the receiver that a client sent a bad request."""
        
        # Subclasses can override this to send an error to the client,
        # if their protocol supports it.  The default action is to
        # close the connection because clients otherwise hang while
        # waiting for the result on the socket.
        
        self.badConnection(socket, address, "bad request: %s" % msg)
        return False


    def requestError(self, socket, address, msg):
        self.badConnection(socket, address, "error: %s" % msg)
        return


    def badConnection(self, socket, address, msg):
        self._info.log("closing connection from [%d@%s]: %s" % (
            address[1], address[0], msg))
        socket.close()
        return


    def __init__(self, name=None):
        Service.__init__(self, name)
        return


    def _createPortMonitor(self):
        import pyre.ipc
        return pyre.ipc.monitor('tcp')


# version
__id__ = "$Id: TCPService.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
