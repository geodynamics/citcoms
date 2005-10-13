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


def main():


    from pyre.applications.Script import Script


    class PulseApp(Script):


        class Inventory(Script.Inventory):

            import pulse
            import pyre.inventory

            solver = pyre.inventory.facility('solver', factory=pulse.solver)


        def main(self, *args, **kwds):
            self.inventory.solver.dump()
            return


        def __init__(self):
            Script.__init__(self, 'pulse')
            return


    app = PulseApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: solver.py,v 1.1.1.1 2005/03/08 16:13:57 aivazis Exp $"

# End of file 
