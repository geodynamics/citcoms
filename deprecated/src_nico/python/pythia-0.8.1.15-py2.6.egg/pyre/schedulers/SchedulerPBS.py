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


from BatchScheduler import BatchScheduler
import os, sys


class SchedulerPBS(BatchScheduler):
    
    
    name = "pbs"
    

    import pyre.inventory as pyre
    
    command       = pyre.str("command", default="qsub")
    qsubOptions   = pyre.list("qsub-options")
    resourceList  = pyre.list("resource-list")
    ppn           = pyre.int("ppn", default=1)
    
    
    def schedule(self, job):
        import pyre.util as util

        # Fix-up the job.
        if not job.task:
            job.task = "jobname"
        job.walltime = util.hms(job.dwalltime.value)
        job.arguments = ' '.join(job.arguments)
        job.resourceList = self.buildResourceList(job)
        
        # Generate the main PBS batch script.
        script = self.retrieveTemplate('batch.sh', ['schedulers', 'scripts', self.name])
        if script is None:
            self._error.log("could not locate batch script template for '%s'" % self.name)
            sys.exit(1)
        
        script.scheduler = self
        script.job = job
        
        if self.dry:
            print script
            return

        try:
            import os
            from popen2 import Popen4

            cmd = [self.command]
            self._info.log("spawning: %s" % ' '.join(cmd))
            child = Popen4(cmd)
            self._info.log("spawned process %d" % child.pid)

            print >> child.tochild, script
            child.tochild.close()

            for line in child.fromchild:
                self._info.line("    " + line.rstrip())
            status = child.wait()
            self._info.log()

            exitStatus = None
            if (os.WIFSIGNALED(status)):
                statusStr = "signal %d" % os.WTERMSIG(status)
            elif (os.WIFEXITED(status)):
                exitStatus = os.WEXITSTATUS(status)
                statusStr = "exit %d" % exitStatus
            else:
                statusStr = "status %d" % status
            self._info.log("%s: %s" % (cmd[0], statusStr))
        
        except IOError, e:
            self._error.log("%s: %s" % (self.command, e))
            return
        
        if exitStatus == 0:
            pass
        else:
            sys.exit("%s: %s: %s" % (sys.argv[0], cmd[0], statusStr))
        
        return


    def buildResourceList(self, job):

        resourceList = self.resourceList
        if self.ppn:
            resourceList.append(
                "nodes=%d:ppn=%d" % ((job.nodes / self.ppn) + (job.nodes % self.ppn and 1 or 0), self.ppn)
                )
        else:
            resourceList.append(
                "nodes=%d" % job.nodes
                )

        walltime = job.walltime
        if max(walltime):
            resourceList.append("walltime=%d:%02d:%02d" % walltime)

        return resourceList


    def jobId(cls):
        return os.environ['PBS_JOBID']
    jobId = classmethod(jobId)


# end of file 
