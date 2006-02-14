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


    class MonitorApp(Script):


        class Inventory(Script.Inventory):

            import pyre.inventory
            from pyre.units.SI import second

            mode = pyre.inventory.str('mode', default='tcp')
            port = pyre.inventory.str('port', default=50000)
            maxTimeouts = pyre.inventory.str('maxTimeouts', default=10)
            timeout = pyre.inventory.dimensional('timeout', default=1*second)


        def main(self, *args, **kwds):
            import pyre.ipc

            print "creating selector"
            selector = pyre.ipc.selector()
            
            print "creating port mointor in mode '%s'" % self.inventory.mode
            monitor = pyre.ipc.monitor(mode=self.inventory.mode)

            monitor.install(self.inventory.port)
            print "monitor installed on port %d" % monitor.port

            print "registering callbacks"
            selector.notifyWhenIdle(self.onTimeout)
            selector.notifyOnReadReady(monitor, self.onConnectionAttempt)

            print "entering event loop"
            self._count = 0
            selector.watch(self.inventory.timeout.value)

            return


        def onTimeout(self, selector):
            print "timeout %4d" % self._count
            self._count += 1
            return self._count < self.inventory.maxTimeouts


        def onConnectionAttempt(self, selector, fd):
            fd.close()
            print "connection attempt: selector=%r, fd=%r" % (selector, fd)
            return False


        def __init__(self):
            Script.__init__(self, 'monitor')
            self._count = 0
            return


    app = MonitorApp()
    return app.run()


# main
if __name__ == '__main__':
    # invoke the application shell
    main()


# version
__id__ = "$Id: monitor.py,v 1.1.1.1 2005/03/08 16:13:49 aivazis Exp $"

# End of file 
