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
    from pyre.applications.DynamicComponentHarness import DynamicComponentHarness


    class Harness(Script, DynamicComponentHarness):


        def main(self, *args, **kwds):
            component = self.harnessComponent()
            if not component:
                print "ooops!"
                return

            curator = self.getCurator()
            weaver = self.weaver
            renderer = curator.codecs['pml'].renderer
            weaver.renderer = renderer

            configuration = self.component.retrieveConfiguration()
            print "\n".join(weaver.render(configuration))

            return


        def __init__(self, name):
            Script.__init__(self, name)
            DynamicComponentHarness.__init__(self)
            return


        def _defaults(self):
            Script._defaults(self)
            self.inventory.typos = 'relaxed'
            return


    app = Harness('harness')
    app.run()


# main
if __name__ == '__main__':
    main()


# version
__id__ = "$Id: harness.py,v 1.1.1.1 2005/03/08 16:13:50 aivazis Exp $"

# End of file 
