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


from Launcher import Launcher


class LauncherMPICH(Launcher):


    class Inventory(Launcher.Inventory):

        import pyre.inventory

        dry = pyre.inventory.bool("dry", default=False)
        debug = pyre.inventory.bool("debug", default=False)
        nodegen = pyre.inventory.str("nodegen")
        extra = pyre.inventory.str("extra")
        command = pyre.inventory.str("command", default="mpirun")
        python_mpi = pyre.inventory.str("python-mpi", default="`which mpipython.exe`")
        machinefile = pyre.inventory.str("machinefile", default="mpirun.nodes")


    def launch(self):
        args = self._buildArgumentList()
        if not args:
            return False
        
        command = " ".join(args)
        self._info.log("executing: {%s}" % command)

        dry = self.inventory.dry
        if not dry:
            import os
            os.system(command)
            return True

        return False

            
    def __init__(self):
        Launcher.__init__(self, "mpirun")
        return


    def _buildArgumentList(self):
        import sys

        nodes = self.nodes
        nodelist = self.nodelist
        nodegen = self.inventory.nodegen
        python_mpi = self.inventory.python_mpi
        machinefile = self.inventory.machinefile

        if not nodes:
            nodes = len(nodelist)

        if nodes < 2:
            self.inventory.nodes = 1
            return []
        
        # build the command
        args = []
        args.append(self.inventory.command)
        args.append(self.inventory.extra)
        args.append("-np %d" % nodes)

        # use only the specific nodes specified explicitly
        if nodelist:
            file = open(machinefile, "w")
            for node in nodelist:
                file.write((nodegen + '\n') % node)
            file.close()
            args.append("-machinefile %s" % machinefile)

        # add the parallel version of the interpreter on the command line
        args.append(python_mpi)

        args += sys.argv
        args.append("--mode=worker")

        return args


# version
__id__ = "$Id: LauncherMPICH.py,v 1.1.1.1 2005/03/08 16:13:30 aivazis Exp $"

# End of file 
