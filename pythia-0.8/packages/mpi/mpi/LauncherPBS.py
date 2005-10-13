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

# NYI: this needs some serious attention

from Launcher import Launcher


class LauncherPBS(Launcher):


    class Inventory(Launcher.Inventory):

        import pyre.inventory

        inventory = [
            pyre.properties.bool("dry", False),
            pyre.properties.bool("debug", False),
            pyre.properties.str("extra"),

            pyre.properties.str("command", "qsub"),
            pyre.properties.str("task"),

            pyre.properties.str("wallclock"),
            pyre.properties.bool("mail", False),
            pyre.properties.int("nodes", 1),
            pyre.properties.str("queue", "standard"),

            pyre.properties.timerep("script"),
            pyre.properties.str("stdout"),
            pyre.properties.str("stderr"),

            pyre.properties.str("ext_script", ".pbs"),
            pyre.properties.str("ext_stdout", ".stdout"),
            pyre.properties.str("ext_stderr", ".stderr"),
            ]


    def launch(self):
        import sys

        dry = self.inventory.dry

##FELDMANN
        nodes = self.inventory.nodes
        nodelist = self.inventory.nodelist
        nodegen = self.inventory.nodegen
        python_mpi = self.inventory.python_mpi
        machinefile = self.inventory.machinefile

        if not nodes:
            nodes = len(nodelist)

        if nodes < 2:
            self.inventory.nodes = 1
            return False
##FELDMANN

        
        # build the command
        args = []
        args.append(self.inventory.command)
        args.append(self.inventory.extra)

        args.append(self.inventory.scriptName)
        scriptFile = open(self.inventory.scriptName,"w")
        scriptFile.writeline(build_script_string())
        scriptFile.close()
        
        args += sys.argv

        command = " ".join(args)
        self._info.log("executing: {%s}" % command)

        if not dry:
            self._execStrategy(command)

        return True

            
    def build_script_string(self):
        script  = [
           "", 
           ]

        queue = self.inventory.queue
        if queue:
            script.append("#PBS -q %s" % queue)

        task = self.inventory.task
        if task:
            script.append("#PBS -N %s" % task)

        nodes = self.inventory.nodes
        if nodes:
            script.append("#PBS -l nodes %s" % nodes)

        clock = self.inventory.wallclock
        if clock:
            script.append("#PBS -l walltime %s" % clock)

        script += [
            "#PBS -o %s" % pyre.inventory.stdout,
            "#PBS -e %s" % pyre.inventory.stderr,
            ]

        # add the mpirun command line
        # script += "some command line to submit\n\n"

        return script


    def __init__(self):
        Launcher.__init__(self, "mpirun")
        return


# version
__id__ = "$Id: LauncherPBS.py,v 1.1.1.1 2005/03/08 16:13:29 aivazis Exp $"

# End of file 
