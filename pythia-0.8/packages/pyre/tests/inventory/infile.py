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

            input = pyre.inventory.inputFile('input')


        def main(self, *args, **kwds):
            print self.inventory.input
            for line in self.inventory.input:
                print line,
            return


        def __init__(self):
            Script.__init__(self, 'file')
            self.input = ''
            return


        def _defaults(self):
            Script._defaults(self)
            return


        def _configure(self):
            Script._configure(self)
            return


        def _init(self):
            Script._init(self)
            self.input = self.inventory.input
            return


    app = FileApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: infile.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

# End of file 
