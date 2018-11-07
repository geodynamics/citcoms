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


class FormHiddenInput(Element):


    def identify(self, inspector):
        return inspector.onFormHiddenInput(self)


    def __init__(self, name, value, **kwds):
        Element.__init__(self, tag='input', name=name, value=value, type='hidden', **kwds)
        return

# version
__id__ = "$Id: FormHiddenInput.py,v 1.1 2005/03/27 15:13:50 aivazis Exp $"

# End of file 
