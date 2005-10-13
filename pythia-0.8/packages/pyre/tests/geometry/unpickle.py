#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                              Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class UnpickleScript(Script):


    class Inventory(Script.Inventory):

        import pyre.geometry
        import pyre.inventory

        name = pyre.inventory.str("name", default="geometry-test.pml")
        modeller = pyre.inventory.facility("modeller", factory=pyre.geometry.loader)


    def main(self, *args, **kwds):
        bodies = self.modeller.model()

        index = 0
        for body in bodies:
            index += 1
            print "body %d: {%s}" % (index, body)

        return


    def __init__(self):
        Script.__init__(self, "unpickle")
        self.modeller = None
        return


    def _configure(self):
        self.modeller = self.inventory.modeller
        self.modeller.source = self.inventory.name
        return


# main
if __name__ == "__main__":
    app = UnpickleScript()
    app.run()


# version
__id__ = "$Id: unpickle.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

#
# End of file
