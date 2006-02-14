#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def main():


    from opal.applications.CGI import CGI


    class HelloApp(CGI):

        class Inventory(CGI.Inventory):

            import pyre.inventory

            name = pyre.inventory.str("name", default="world")
            name.meta['tip'] = "the target of the greeting"


        def main(self):
            import os
            pid = os.getpid()
            euid = os.geteuid()
            uid = os.getuid()
            
            print '<pre>'
            print "(%d, %s): Hello %s!" % (pid, uid, self.inventory.name)
            print '</pre>'
            return


        def __init__(self):
            CGI.__init__(self, 'hello')
            return


    import journal
    journal.info('opal.cmdline').activate()
    journal.debug('opal.commandline').activate()

    app = HelloApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: hello.py,v 1.1.1.1 2005/03/15 06:09:10 aivazis Exp $"

# End of file 
