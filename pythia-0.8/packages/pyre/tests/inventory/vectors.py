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


    class VectorsApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            seq = pyre.inventory.list('seq')
            seq.meta['tip'] = 'the list inventory item'

            from pyre.units.length import m
            vec = pyre.inventory.dimensional('vec', default=(0*m, 0*m, 0*m))
            vec.meta['tip'] = 'the vector inventory item'


        def main(self, *args, **kwds):
            print "seq:", self.inventory.seq
            print "vec:", self.inventory.vec
            return


        def __init__(self):
            Script.__init__(self, 'vectors')
            return


    app = VectorsApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: vectors.py,v 1.1 2005/03/24 01:56:34 aivazis Exp $"

# End of file 
