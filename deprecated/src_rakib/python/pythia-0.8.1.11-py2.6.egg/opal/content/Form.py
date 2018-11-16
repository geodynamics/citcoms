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


from ElementContainer import ElementContainer
from ParagraphFactory import ParagraphFactory


class Form(ElementContainer, ParagraphFactory):


    def identify(self, inspector):
        return inspector.onForm(self)


    def control(self, **kwds):
        from FormControl import FormControl
        control = FormControl(**kwds)
        self.contents.append(control)
        return control


    def field(self, **kwds):
        from FormField import FormField
        field = FormField(**kwds)
        self.contents.append(field)
        return field


    def hidden(self, **kwds):
        from FormHiddenInput import FormHiddenInput
        field = FormHiddenInput(**kwds)
        self.contents.append(field)
        return field


    def password(self, **kwds):
        from Input import Input
        control = Input(type="password", **kwds)
        
        from FormField import FormField
        field = FormField(control)
        self.contents.append(field)

        return control


    def text(self, **kwds):
        from Input import Input
        control = Input(**kwds)
        
        from FormField import FormField
        field = FormField(control)
        self.contents.append(field)

        return control


    def __init__(self, name, action, legend=None, **kwds):
        ElementContainer.__init__(self, 'form', name=name, action=action, method="post", **kwds)
        ParagraphFactory.__init__(self)
        
        self.legend = legend
        return

# version
__id__ = "$Id: Form.py,v 1.4 2005/05/05 04:44:27 pyre Exp $"

# End of file 
