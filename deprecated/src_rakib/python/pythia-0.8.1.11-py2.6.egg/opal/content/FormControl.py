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


class FormControl(Element):


    def identify(self, inspector):
        return inspector.onFormControl(self)


    def __init__(self, name, type, value, **kwds):
        Element.__init__(self, 'div', cls='formControls', **kwds)

        self.name = name
        self.type = type
        self.value = value

        return

# version
__id__ = "$Id: FormControl.py,v 1.2 2005/03/27 15:17:19 aivazis Exp $"

# End of file 
