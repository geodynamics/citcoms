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


    class LogicalsApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            negative = pyre.inventory.float('negative', default=-1.0)
            negative.validator = pyre.inventory.isNot(pyre.inventory.greaterEqual(0.0))

            scale = pyre.inventory.float('scale', default=0.0)
            scale.validator = pyre.inventory.isBoth(
                pyre.inventory.greaterEqual(0.0), pyre.inventory.lessEqual(1.0))

            outside = pyre.inventory.float('outside', default=100.0)
            outside.validator = pyre.inventory.isEither(
                pyre.inventory.less(0.0), pyre.inventory.greater(1.0))


        def main(self, *args, **kwds):
            print 'scale:', self.inventory.scale
            print 'negative:', self.inventory.negative
            print 'outside:', self.inventory.outside
            return


        def __init__(self):
            Script.__init__(self, 'logicals')
            return


    app = LogicalsApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: logicals.py,v 1.1 2005/03/10 04:04:37 aivazis Exp $"

# End of file 
