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


from Session import Session


class TCPSession(Session):


    from RequestError import RequestError


    def request(self, command, args=None):
        if args is None:
            args = ()

        import pyre.services
        request = pyre.services.request(command, args)

        self._info.log("sending request: command=%r" % command)
        self.marshaller.send(request, self._connection)
        self._info.log("request sent")

        result = self.marshaller.receive(self._connection)
        return result


    def __init__(self, name):
        Session.__init__(self, name, protocol='tcp')
        return


    def _init(self):
        try:
            self._connect()
        except self._connection.ConnectionError, error:
            raise self.RequestError(str(error))

        return


# version
__id__ = "$Id: TCPSession.py,v 1.2 2005/03/14 22:55:35 aivazis Exp $"

# End of file 
