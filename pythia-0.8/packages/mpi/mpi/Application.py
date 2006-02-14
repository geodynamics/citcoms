#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class Application(Script):


    class Inventory(Script.Inventory):

        import pyre.inventory

        from LauncherMPICH import LauncherMPICH

        mode = pyre.inventory.str(
            name="mode", default="server", validator=pyre.inventory.choice(["server", "worker"]))
        launcher = pyre.inventory.facility("launcher", factory=LauncherMPICH)


    def execute(self, *args, **kwds):

        if self.inventory.mode == "worker":
            self.onComputeNodes(*args, **kwds)
            return
        
        self.onServer(*args, **kwds)

        return


    def onComputeNodes(self, *args, **kwds):
        self.main(*args, **kwds)
        return


    def onServer(self, *args, **kwds):
        self._debug.log("%s: onServer" % self.name)

        launcher = self.inventory.launcher
        launched = launcher.launch()
        if not launched:
            self.onComputeNodes(*args, **kwds)
        
        return


    def __init__(self, name):
        Script.__init__(self, name)
        self.launcher = None
        return


    def _configure(self):
        Script._configure(self)
        self.launcher = self.inventory.launcher
        return


# version
__id__ = "$Id: Application.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file 
