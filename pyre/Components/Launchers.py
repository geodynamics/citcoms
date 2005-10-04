#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
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
# MPI Launchers:
#     (Replacement) Launcher for MPICH
#     Launcher for LAM/MPI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


class LauncherMPI(Launcher):


    class Inventory(Launcher.Inventory):

        import pyre.inventory

        dry = pyre.inventory.bool("dry", default=False)
        dry.meta['tip'] = "prints the command line and exits"
        
        debug = pyre.inventory.bool("debug", default=False)

        Launcher.Inventory.nodes.meta['tip'] = """number of machine nodes"""
        Launcher.Inventory.nodelist.meta['tip'] = """a comma-separated list of machine names in square brackets (e.g., [101-103,105,107])"""
        nodegen = pyre.inventory.str("nodegen")
        nodegen.meta['tip'] = """a printf-style format string, used in conjunction with 'nodelist' to generate the list of machine names (e.g., "n%03d")"""
        
        extra = pyre.inventory.str("extra")
        extra.meta['tip'] = "extra arguments to pass to mpirun"
        
        command = pyre.inventory.str("command", default="mpirun")
        python_mpi = pyre.inventory.str("python-mpi", default=which("mpipython.exe"))
        re_exec = pyre.inventory.bool("re-exec", default=True)


    def launch(self):
        args = self._buildArgumentList()
        if not args:
            return self.inventory.dry
        
        command = " ".join(args)
        self._info.log("executing: {%s}" % command)

        if self.inventory.dry:
            print command
            return True
        
        import os
        os.system(command)
        return True

            
    def _buildArgumentList(self):
        import sys

        python_mpi = self.inventory.python_mpi

        if not self.nodes:
            self.nodes = len(self.nodelist)

        if self.nodes < 2:
            import mpi
            if mpi.world().handle():
                self.inventory.nodes = 1
                return []
            elif self.inventory.re_exec:
                # re-exec under mpipython.exe
                args = []
                args.append(python_mpi)
                args += sys.argv
                args.append("--launcher.re-exec=False") # protect against infinite regress
                return args
            else:
                # We are under the 'mpipython.exe' interpreter,
                # yet the 'mpi' module is non-functional.  Attempt to
                # re-raise the exception that may have caused this.
                import mpi._mpi
                return []
        
        # build the command
        args = []
        args.append(self.inventory.command)
        self._appendMpiRunArgs(args)

        # add the parallel version of the interpreter on the command line
        args.append(python_mpi)

        args += sys.argv
        args.append("--mode=worker")

        return args

    
    def _appendMpiRunArgs(self, args):
        args.append(self.inventory.extra)
        args.append("-np %d" % self.nodes)
        
        # use only the specific nodes specified explicitly
        if self.nodelist:
            self._appendNodeListArgs(args)


class LauncherMPICH(LauncherMPI):


    class Inventory(LauncherMPI.Inventory):

        import pyre.inventory

        machinefile = pyre.inventory.str("machinefile", default="mpirun.nodes")
        machinefile.meta['tip'] = """filename of machine file"""


    def __init__(self):
        LauncherMPI.__init__(self, "mpich")


    def _appendNodeListArgs(self, args):
        machinefile = self.inventory.machinefile
        nodegen = self.inventory.nodegen
        file = open(machinefile, "w")
        for node in self.nodelist:
            file.write((nodegen + '\n') % node)
        file.close()
        args.append("-machinefile %s" % machinefile)


class LauncherLAMMPI(LauncherMPI):


    class Inventory(LauncherMPI.Inventory):

        import pyre.inventory

        environment = pyre.inventory.list("environment", default=defaultEnvironment)
        environment.meta['tip'] = """a comma-separated list of environment variables to export to the batch job"""


    def __init__(self):
        LauncherMPI.__init__(self, "lam-mpi")


    def _appendMpiRunArgs(self, args):
        args.append("-x %s" % ','.join(self.inventory.environment))
        super(LauncherLAMMPI, self)._appendMpiRunArgs(args)


    def _appendNodeListArgs(self, args):
        nodegen = self.inventory.nodegen
        args.append("n" + ",".join([(nodegen) % node for node in self.nodelist]))


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

        # Ignore 'nodegen' so that the examples will work without modification.
        nodegen = pyre.inventory.str("nodegen")
        nodegen.meta['tip'] = """(ignored)"""

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
        if self.inventory.dry:
            print self._buildScript()
            print "# submit with:"
            print "#", self._buildBatchCommand()
            return True
        
        # write the script
        scriptFile = open(expandMacros("${script}", self.inv), "w")
        scriptFile.write(self._buildScript())
        scriptFile.close()

        # build the batch command
        command = self._buildBatchCommand()
        self._info.log("executing: {%s}" % command)

        import os
        os.system(command)
        return True


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
        
        command = pyre.inventory.str("command", default="mpijob mpirun")
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
            Script.__init__(self, "CitcomS")

    
    app = TestApp()
    app.run()


# version
__id__ = "$Id$"

# End of file 
