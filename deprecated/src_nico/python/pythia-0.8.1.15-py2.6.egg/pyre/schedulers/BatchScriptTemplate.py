#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.components import Component


class BatchScriptTemplate(Component):


    """base class for batch script templates"""


    def __init__(self, name):
        Component.__init__(self, name, facility='template')
        self.scheduler = None
        self.script = None


    def render(self):
        raise NotImplementedError("class %r must override 'render'" % self.__class__.__name__)


    def __str__(self):
        return self.render()


# end of file 
