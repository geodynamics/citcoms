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


    class FileApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory

            output = pyre.inventory.outputFile('output')


        def main(self, *args, **kwds):
            self.dump()
            
            print self.inventory.output
            print >> self.inventory.output, 'Hello world!'
            return


        def __init__(self):
            Script.__init__(self, 'file')
            self.output = ''
            return


        def _defaults(self):
            Script._defaults(self)
            return


        def _configure(self):
            Script._configure(self)
            return


        def _init(self):
            Script._init(self)
            self.output = self.inventory.output
            return


    app = FileApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: outfile.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
