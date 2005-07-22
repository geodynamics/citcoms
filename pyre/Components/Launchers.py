#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#=====================================================================
#
#                             CitcomS.py
#                 ---------------------------------
#
#                              Authors:
#            Eh Tan, Eun-seo Choi, and Pururav Thoutireddy 
#          (c) California Institute of Technology 2002-2005
#
#        By downloading and/or installing this software you have
#       agreed to the CitcomS.py-LICENSE bundled with this software.
#             Free for non-commercial academic research ONLY.
#      This program is distributed WITHOUT ANY WARRANTY whatsoever.
#
#=====================================================================
#
#  Copyright June 2005, by the California Institute of Technology.
#  ALL RIGHTS RESERVED. United States Government Sponsorship Acknowledged.
# 
#  Any commercial use must be negotiated with the Office of Technology
#  Transfer at the California Institute of Technology. This software
#  may be subject to U.S. export control laws and regulations. By
#  accepting this software, the user agrees to comply with all
#  applicable U.S. export laws and regulations, including the
#  International Traffic and Arms Regulations, 22 C.F.R. 120-130 and
#  the Export Administration Regulations, 15 C.F.R. 730-744. User has
#  the responsibility to obtain export licenses, or other export
#  authority as may be required before exporting such information to
#  foreign countries or providing access to foreign nationals.  In no
#  event shall the California Institute of Technology be liable to any
#  party for direct, indirect, special, incidental or consequential
#  damages, including lost profits, arising out of the use of this
#  software and its documentation, even if the California Institute of
#  Technology has been advised of the possibility of such damage.
# 
#  The California Institute of Technology specifically disclaims any
#  warranties, including the implied warranties or merchantability and
#  fitness for a particular purpose. The software and documentation
#  provided hereunder is on an "as is" basis, and the California
#  Institute of Technology has no obligations to provide maintenance,
#  support, updates, enhancements or modifications.
#
#=====================================================================
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from mpi.Launcher import Launcher
from pyre.util import expandMacros


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# utility functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def which(filename):
    from os.path import abspath, exists, join
    from os import environ, pathsep
    dirs = environ['PATH'].split(pathsep)
    for dir in dirs:
       pathname = join(dir, filename)
       if exists(pathname):
           return abspath(pathname)
    return filename


def hms(t):
    return (int(t / 3600), int((t % 3600) / 60), int(t % 60))


defaultEnvironment = "[EXPORT_ROOT,LD_LIBRARY_PATH,PYTHONPATH]"

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Launcher for LAM/MPI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LauncherLAMMPI(Launcher):


    class Inventory(Launcher.Inventory):

        import pyre.inventory

        dry = pyre.inventory.bool("dry", default=False)
        debug = pyre.inventory.bool("debug", default=False)
        nodegen = pyre.inventory.str("nodegen")
        extra = pyre.inventory.str("extra")
        command = pyre.inventory.str("command", default="mpirun")
        python_mpi = pyre.inventory.str("python-mpi", default=which("mpipython.exe"))
        environment = pyre.inventory.list("environment", default=defaultEnvironment)


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
        Launcher.__init__(self, "lam-mpi")
        return


    def _buildArgumentList(self):
        import sys

        nodes = self.nodes
        nodelist = self.nodelist
        nodegen = self.inventory.nodegen
        python_mpi = self.inventory.python_mpi

        if not nodes:
            nodes = len(nodelist)

        if nodes < 2:
            self.inventory.nodes = 1
            return []
        
        # build the command
        args = []
        args.append(self.inventory.command)
        args.append(self.inventory.extra)
        args.append("-x %s" % ','.join(self.inventory.environment))
        args.append("-np %d" % nodes)

        # use only the specific nodes specified explicitly
        if nodelist:
            args.append("n" + ",".join([(nodegen) % node for node in nodelist]))

        # add the parallel version of the interpreter on the command line
        args.append(python_mpi)

        args += sys.argv
        args.append("--mode=worker")

        return args


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# These are Launchers for batch schedulers found on the TeraGrid.
# These should be incorporated into Pythia eventually.

# With something like StringTemplate by Terence Parr and Marq Kole,
# the batch scripts could be generated entirely from an
# inventory-data-driven template.

#     http://www.stringtemplate.org/doc/python-doc.html

# This code uses a hybrid approach, mixing Python logic with primitive
# templates powered by pyre.util.expandMacros().
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LauncherBatch(Launcher):


    class Inventory(Launcher.Inventory):

        import os, sys
        import pyre.inventory
        from pyre.units.time import minute

        dry = pyre.inventory.bool("dry", default=False)
        debug = pyre.inventory.bool("debug", default=False)

        python_mpi = pyre.inventory.str("python-mpi", default=which("mpipython.exe"))
        task = pyre.inventory.str("task")

        walltime = pyre.inventory.dimensional("walltime", default=30.0*minute)
        mail = pyre.inventory.bool("mail", default=False)
        queue = pyre.inventory.str("queue")

        directory = pyre.inventory.str("directory", default="${cwd}")
        script = pyre.inventory.str("script", default="${directory}/script")
        stdout = pyre.inventory.str("stdout", default="${directory}/stdout")
        stderr = pyre.inventory.str("stderr", default="${directory}/stderr")
        environment = pyre.inventory.list("environment", default=defaultEnvironment)

        cwd = pyre.inventory.str("cwd", default=os.getcwd())
        argv = pyre.inventory.str("argv", default=(' '.join(sys.argv)))


    def launch(self):

        # write the script
        scriptFile = open(expandMacros("${script}", self.inv), "w")
        scriptFile.write(self._buildScript())
        scriptFile.close()

        # build the batch command
        command = self._buildBatchCommand()
        self._info.log("executing: {%s}" % command)

        dry = self.inventory.dry
        if not dry:
            import os
            os.system(command)
            return True

        return False


    def __init__(self, name):
        Launcher.__init__(self, name)
        
        # Used to recursively expand ${macro) in format strings using my inventory.
        class InventoryAdapter(object):
            def __init__(self, launcher):
                self.launcher = launcher
            def __getitem__(self, key):
                return expandMacros(str(self.launcher.inventory.getTraitValue(key)), self)
        self.inv = InventoryAdapter(self)
        
        return


    def _buildBatchCommand(self):
        return expandMacros("${batch-command} ${script}", self.inv)


    def _buildScript(self):
        script = [
            "#!/bin/sh",
            ]
        self._buildScriptDirectives(script)
        script += [
            expandMacros('''\

cd ${directory}
${command} ${python-mpi} ${argv} --mode=worker
''', self.inv)
            ]
        script = "\n".join(script) + "\n"
        return script


# Note: mpi.LauncherPBS in Pythia-0.8 does not work!

class LauncherPBS(LauncherBatch):


    class Inventory(LauncherBatch.Inventory):
        
        import pyre.inventory
        
        command = pyre.inventory.str("command", default="mpirun -np ${nodes} -machinefile $PBS_NODEFILE") # Sub-launcher?
        batch_command = pyre.inventory.str("batch-command", default="qsub")


    def __init__(self):
        LauncherBatch.__init__(self, "pbs")


    def _buildScriptDirectives(self, script):
        
        queue = self.inventory.queue
        if queue:
            script.append("#PBS -q %s" % queue)

        task = self.inventory.task
        if task:
            script.append("#PBS -N %s" % task)

        if self.inventory.stdout:
            script.append(expandMacros("#PBS -o ${stdout}", self.inv))
        if self.inventory.stderr:
            script.append(expandMacros("#PBS -e ${stderr}", self.inv))

        resourceList = self._buildResourceList()

        script += [
            "#PBS -V", # export qsub command environment to the batch job
            "#PBS -l %s" % resourceList,
            ]

        return script



    def _buildResourceList(self):

        resourceList = [
            "nodes=%d" % self.nodes,
            ]

        walltime = self.inventory.walltime.value
        if walltime:
            resourceList.append("walltime=%d:%02d:%02d" % hms(walltime))

        resourceList = ",".join(resourceList)

        return resourceList


class LauncherLSF(LauncherBatch):


    class Inventory(LauncherBatch.Inventory):
        
        import pyre.inventory
        
        command = pyre.inventory.str("command", default="pam -g 1 gmmpirun_wrapper") # TACC-specific?
        batch_command = pyre.inventory.str("batch-command", default="bsub")


    def __init__(self):
        LauncherBatch.__init__(self, "lsf")


    def _buildBatchCommand(self):
        return expandMacros("${batch-command} < ${script}", self.inv)


    def _buildScriptDirectives(self, script):

        # LSF scripts must have a job name; otherwise strange "/bin/sh: Event not found"
        # errors occur (tested on TACC's Lonestar system).
        task = self.inventory.task
        if not task:
            task = "jobname"
        script.append("#BSUB -J %s" % task)
        
        queue = self.inventory.queue
        if queue:
            script.append("#BSUB -q %s" % queue)

        walltime = self.inventory.walltime.value
        if walltime:
            script.append("#BSUB -W %d:%02d" % hms(walltime)[0:2])
        
        if self.inventory.stdout:
            script.append(expandMacros("#BSUB -o ${stdout}", self.inv))
        if self.inventory.stderr:
            script.append(expandMacros("#BSUB -e ${stderr}", self.inv))
            
        script += [
            "#BSUB -n %d" % self.nodes,
            ]

        return script


class LauncherGlobus(LauncherBatch):


    class Inventory(LauncherBatch.Inventory):

        import pyre.inventory

        batch_command = pyre.inventory.str("batch-command", default="globusrun")
        resource = pyre.inventory.str("resource", default="localhost")


    def _buildBatchCommand(self):
        return expandMacros("${batch-command} -b -r ${resource} -f ${script}", self.inv)


    def __init__(self):
        LauncherBatch.__init__(self, "globus")
        return


    def _buildScript(self):
        import sys

        script = [
            expandMacros('''\
&   (jobType=mpi)
    (executable="${python-mpi}")
    (count=${nodes})
    (directory="${directory}")
    (stdout="${stdout}")
    (stderr="${stderr}")''', self.inv),
            ]
        
        script.append('    (environment = %s)' % self._buildEnvironment())

        # add the arguments
        args = sys.argv
        args.append("--mode=worker")
        command = '    (arguments= ' + ' '.join([('"%s"' % arg) for arg in args]) + ')'
        script.append(command)

        script = '\n'.join(script) + '\n'

        return script


    def _buildEnvironment(self):
        from os import environ
        #vars = environ.keys()
        vars = self.inventory.environment
        env = [('(%s "%s")' % (var, environ.get(var,""))) for var in vars]
        env = ' '.join(env)
        return env


# main
if __name__ == "__main__":

    
    from pyre.applications.Script import Script

    
    class TestApp(Script):

        
        class Inventory(Script.Inventory):
            
            import pyre.inventory
            from mpi.Launcher import Launcher

            launcher = pyre.inventory.facility("launcher", default="mpich")


        def main(self, *args, **kwds):
            launcher = self.inventory.launcher
            if launcher:
                try:
                    # batch launcher
                    print launcher._buildScript()
                    print
                    print "# submit with", launcher._buildBatchCommand()
                except AttributeError:
                    # direct launcher
                    print ' '.join(launcher._buildArgumentList())
            return

        
        def __init__(self):
            Script.__init__(self, "citcoms")

    
    app = TestApp()
    app.run()


# version
__id__ = "$Id: Launchers.py,v 1.1.2.1 2005/07/22 03:04:51 leif Exp $"

# End of file 
