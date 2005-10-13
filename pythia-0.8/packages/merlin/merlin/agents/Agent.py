#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components.Component import Component


class Agent(Component):


    def execute(self, merlin, project):
        raise NotImplementedError("class '%s' must override 'execute'" % self.__class__.__name__)


    def __init__(self, name, action):
        Component.__init__(self, name, action)
        return


# version
__id__ = "$Id: Agent.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
