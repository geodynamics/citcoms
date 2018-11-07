#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2004  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from Element import Element


class Selector(Element):


    def identify(self, inspector):
        return inspector.onSelector(self)


    def __init__(self, name, entries, label, selected=None, help='',  **kwds):
        Element.__init__(self, tag='select', name=name, **kwds)

        self.label = label
        self.help = help
        self.entries = entries
        self.selection = selected

        return
        

# version
__id__ = "$Id: Selector.py,v 1.1 2005/04/25 05:38:02 pyre Exp $"

# End of file 
