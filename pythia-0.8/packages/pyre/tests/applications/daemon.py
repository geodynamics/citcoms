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

    from pyre.applications.Daemon import Daemon
    from pyre.applications.Script import Script

    class App(Script, Daemon):


        def main(self, *args, **kwds):
            stream = file('/tmp/' + self.name + '.log', 'a')

            import os
            import time
            
            print >> stream, "%d: %s" % (os.getpid(), time.ctime())

            return


        def __init__(self, name):
            Script.__init__(self, name)
            Daemon.__init__(self)
            return


    app = App('daemon')
    return app.run(spawn=False)


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: daemon.py,v 1.2 2005/03/10 21:32:06 aivazis Exp $"

# End of file 
