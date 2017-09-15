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


class Port(object):


    def send(self, string):
        import _mpi
        _mpi.sendString(self._communicator.handle(), self.peer, self.tag, string)
        return


    def receive(self):
        import _mpi
        string = _mpi.receiveString(self._communicator.handle(), self.peer, self.tag)
        return string


    def __init__(self, communicator, peer, tag):
        self.peer = peer
        self.tag = tag
        self._communicator = communicator
        return

# version
__id__ = "$Id: Port.py,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $"

# End of file 
