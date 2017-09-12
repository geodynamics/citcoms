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


from Element import Element


class Input(Element):


    def identify(self, inspector):
        return inspector.onInput(self)


    def __init__(self, id, name, label, help='', type='text', **kwds):
        Element.__init__(self, tag='input', cls="formfield", id=id, name=name, type=type, **kwds)

        self.label = label
        self.help = help

        return

# version
__id__ = "$Id: Input.py,v 1.1 2005/04/25 05:38:18 pyre Exp $"

# End of file 
