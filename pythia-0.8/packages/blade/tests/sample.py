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


from pyre.applications.Script import Script
from pyre.applications.ComponentHarness import ComponentHarness


class SampleApp(Script, ComponentHarness):


    def main(self):
        component = self.harnessComponent()
        component.render()
        return


    def createComponent(self):
        import blade
        return blade.blade()


    def __init__(self, name=None):
        if name is None:
            name = 'sample'
            
        Script.__init__(self, name)
        ComponentHarness.__init__(self)

        return

# main
if __name__ == "__main__":
    app = SampleApp()
    app.run()


# version
__id__ = "$Id: sample.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

# End of file 
