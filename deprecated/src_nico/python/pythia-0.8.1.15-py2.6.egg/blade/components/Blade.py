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


from pyre.components.Component import Component


class Blade(Component):


    class Inventory(Component.Inventory):

        import pyre.inventory

        layout = pyre.inventory.str('layout', default='')
        toolkit = pyre.inventory.str('toolkit', default='')
        language = pyre.inventory.str('language', default='')


    def render(self):

        document = self.retrieveLayout()
        renderer = self.retrieveRenderer()

        return


    def retrieveLayout(self, encoding=None):
        layout = self.layout
        print " ++ layout description in '%s'" % layout

        import os
        base, extension = os.path.splitext(layout)
        if encoding is None:
            if extension:
                encoding = extension[1:]
            else:
                import journal
                journal.error("blade").log("unknown layout type in '%s'" % layout)
                return
            
        print " ++ encoding:", encoding
        try:
            codec = self.codecs[encoding]
        except KeyError:
            import journal
            journal.error("blade").log("unknown encoding '%s'" % encoding)
            return

        print codec.open(base)

        document = None
        return document


    def retrieveRenderer(self, toolkit=None, language=None):
        if toolkit is None:
            toolkit = self.toolkit

        if language is None:
            language = self.language

        renderer = self.retrieveComponent(
            name=language, factory='renderer', args=[self], vault=['toolkits', toolkit])

        return renderer


    def __init__(self, name=None):
        if name is None:
            name = 'blade'

        Component.__init__(self, name, facility="ui")

        self.layout = None
        self.toolkit = None
        self.language = None


        import blade.pml
        self.codecs = {
            'pml': blade.pml.codecPML()
            }

        return


    def _configure(self):
        """transfer the setting to my data members"""
        
        self.layout = self.inventory.layout
        self.toolkit = self.inventory.toolkit
        self.language = self.inventory.language

        return




# version
__id__ = "$Id: Blade.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

# End of file 
