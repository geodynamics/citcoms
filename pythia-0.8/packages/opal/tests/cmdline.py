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


    class CmdlineApp(CGI):


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
            print '</pre>'

            return


        def __init__(self):
            CGI.__init__(self, 'cmdline')
            return


    import journal
    journal.info('opal.cmdline').activate()
    journal.debug('opal.commandline').activate()

    app = CmdlineApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: cmdline.py,v 1.1.1.1 2005/03/15 06:09:10 aivazis Exp $"

# End of file 
