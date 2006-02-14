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


    from opaldemo.applications.WebApplication import WebApplication


    class DemoApp(WebApplication):


        def main(self, *args, **kwds):
            self.render(self.retrievePage("debug"))
            return


        def __init__(self):
            WebApplication.__init__(self, name='main', asCGI=True)
            return


    app = DemoApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: debug.py,v 1.1 2005/03/27 15:13:06 aivazis Exp $"

# End of file 
