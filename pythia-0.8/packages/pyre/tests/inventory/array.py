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


    class ArrayApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            ints = pyre.inventory.array('ints')
            ints.converter = int
            ints.meta['tip'] = 'the array of ints inventory item'

            floats = pyre.inventory.array('floats')
            floats.meta['tip'] = 'the array of floats inventory item'


        def main(self, *args, **kwds):
            print "ints:", self.inventory.ints
            print "floats:", self.inventory.floats
            return


        def __init__(self):
            Script.__init__(self, 'array')
            return


    app = ArrayApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: array.py,v 1.2 2005/04/22 01:29:25 pyre Exp $"

# End of file 
