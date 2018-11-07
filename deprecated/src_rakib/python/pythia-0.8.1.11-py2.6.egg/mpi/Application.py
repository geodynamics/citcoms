#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications.Script import Script


class Application(Script):


    import pyre.inventory
    nodes = pyre.inventory.int("nodes", default=1)

    import pyre.schedulers
    scheduler = pyre.schedulers.scheduler("scheduler", default="none")
    job = pyre.schedulers.job("job")
        
    import pyre.launchers
    launcher = pyre.launchers.facility("launcher", default="mpich")


    nodes.meta['tip'] = """number of machine nodes"""


    def execute(self, *args, **kwds):
        self.onLoginNode(*args, **kwds)


    def onLoginNode(self, *args, **kwds):
        self.scheduleJob(*args, **kwds)


    def scheduleJob(self, *args, **kwds):
        import sys
        
        path = self.pathString()
        requires = "pythia" # ignored -- was "self.requires()"
        entry = self.entryName()
        argv = self.getArgv(*args, **kwds)
        state = self.getStateArgs('launch')
        batchScriptArgs = self.getBatchScriptArgs()
        
        # initialize the job
        job = self.job
        job.nodes = self.nodes
        job.executable = self.jobExecutable
        job.arguments = (["--pyre-start", path, requires,
                          "pyre.schedulers:jobstart"] + batchScriptArgs +
                          [entry] + argv + state)

        # for debugging purposes, add 'mpirun' command as a comment
        launcher = self.prepareLauncher(argv + state)
        job.comments.extend(["[%s] %s" % (launcher.name, comment) for comment in launcher.comments()])

        # schedule
        self.scheduler.schedule(job)
        
        return


    def prepareLauncher(self, argv):
        import sys

        path = self.pathString()
        requires = "pythia" # ignored -- was "self.requires()"
        entry = self.entryName()
        state = self.getStateArgs('compute')
        
        # initialize the launcher
        launcher = self.launcher
        launcher.nodes = self.nodes
        launcher.executable = self.mpiExecutable
        launcher.arguments = ["--pyre-start", path, requires, "mpi:mpistart", entry] + argv + state

        return launcher

    
    def _onLauncherNode(self, *args, **kwds):
        # This method should not be overriden in any application class.
        self.job.id = self.scheduler.jobId()
        self.onLauncherNode(*args, **kwds)


    def onLauncherNode(self, *args, **kwds):
        self.launchParallelRun(*args, **kwds)


    def launchParallelRun(self, *args, **kwds):

        argv = self.getArgv(*args, **kwds)
        launcher = self.prepareLauncher(argv)

        # launch
        launcher.launch()
        
        return


    def _onComputeNodes(self, *args, **kwds):
        # This method should not be overriden in any application class.

        # Don't try this at home.
        argv = kwds['argv']
        for arg in argv:
            if arg.startswith("--macros.job.id="):
                self.job.id = arg.split('=')[1]
                break

        self.onComputeNodes(*args, **kwds)

        return


    def onComputeNodes(self, *args, **kwds):
        self.main(*args, **kwds)


    def getStateArgs(self, stage):
        state = []
        if stage == 'launch':
            state.append("--nodes=%d" % self.nodes) # in case it was computed
            state.append("--macros.nodes=%d" % self.nodes)
        state.extend(self.job.getStateArgs(stage))
        return state


    def getBatchScriptArgs(self):
        SchedulerClass = self.scheduler.__class__
        schedulerClass = SchedulerClass.__module__ + ":" + SchedulerClass.__name__
        return ["--scheduler-class=%s" % schedulerClass]


    def __init__(self, name=None):
        super(Application, self).__init__(name)

        import sys
        from os.path import join, split

        self.executable = sys.executable
        exe = split(self.executable)
        self.jobExecutable = self.executable
        self.mpiExecutable = join(exe[0], "mpi" + exe[1])


# end of file
