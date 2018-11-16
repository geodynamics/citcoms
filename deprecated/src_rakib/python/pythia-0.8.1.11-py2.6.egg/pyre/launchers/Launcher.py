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


class Launcher(Component):
    
    
    import pyre.inventory as pyre
    
    dry         = pyre.bool("dry")
    nodes       = pyre.int("nodes", default=1); nodes.meta['tip'] = """number of machine nodes"""
    nodelist    = pyre.slice("nodelist");
    executable  = pyre.str("executable")
    arguments   = pyre.list("arguments")


    nodelist.meta['tip'] = """a comma-separated list of machine names in square brackets (e.g., [101-103,105,107])"""

    
    def launch(self):
        raise NotImplementedError("class '%s' must override 'launch'" % self.__class__.__name__)


    def argv(self):
        raise NotImplementedError("class '%s' must override 'argv'" % self.__class__.__name__)


    def comments(self):
        return ["command: " + ' '.join(self.argv())]


# end of file 
