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


    class LoadApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            name = pyre.inventory.str('name', default='hello')
            name.meta['tip'] = 'the name of the configuration file to load'

            out = pyre.inventory.outputFile('out', default='out.pml')
            out.meta['tip'] = 'the file in which to save the loaded registry'


        def main(self, *args, **kwds):

            import pyre.inventory
            codec = pyre.inventory.codecPML()
            shelf = codec.open(self.filename)
            registry = shelf['inventory']

            stream = self.out
            document = self.weaver.render(registry)
            print >> stream, "\n".join(document)

            return


        def __init__(self):
            Script.__init__(self, 'load')
            self.filename = ''
            self.out = None
            return


        def _configure(self):
            Script._configure(self)
            self.out = self.inventory.out
            self.filename = self.inventory.name
            return


    app = LoadApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: load.py,v 1.1 2005/03/11 07:07:42 aivazis Exp $"

# End of file 
