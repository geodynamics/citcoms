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


    class LoginApp(CGI):


        class Inventory(CGI.Inventory):

            import pyre.inventory

            username = pyre.inventory.str("username")
            password = pyre.inventory.str("password")


        def main(self):
            import os
            import sys

            print '<pre>'
            print sys.argv
            print '</pre>'

            print '<pre>'
            for key, value in os.environ.iteritems():
                print '    %s = {%s}' % (key, value)
            print '</pre>'

            print '<pre>'
            print self.registry.render()
            print "username:", self.inventory.username
            print "password:", self.inventory.password
            print '</pre>'

            self._info.log("This is an info message")
            self._debug.log("This is a debug message")

            return


        def __init__(self):
            CGI.__init__(self, 'login')
            return


    import journal
    journal.debug('opal.commandline').activate()

    app = LoginApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: login.py,v 1.1.1.1 2005/03/15 06:09:10 aivazis Exp $"

# End of file 
