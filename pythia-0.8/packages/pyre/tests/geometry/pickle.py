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


class PickleScript(Script):


    class Inventory(Script.Inventory):

        import pyre.geometry
        import pyre.inventory

        name = pyre.inventory.str("name", default="geometry-test.pml")
        modeller = pyre.inventory.facility("modeller", default="canister")


    def main(self, *args, **kwds):
        model = self.modeller.model()
        outfile = file(self.inventory.name, "w")

        self.modeller.saveModel(model, outfile)
       
        return


    def __init__(self):
        Script.__init__(self, "pickle")
        self.modeller = None
        return


    def _configure(self):
        Script._configure(self)
        self.modeller = self.inventory.modeller
        return


# main
if __name__ == "__main__":
    app = PickleScript()
    app.run()


# version
__id__ = "$Id: pickle.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

#
# End of file
