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


class SchedulerLSF(BatchScheduler):
    
    
    name = "lsf"
    

    import pyre.inventory as pyre
    
    command      = pyre.str("command", default="bsub")
    bsubOptions  = pyre.list("bsub-options")
    
    
    def schedule(self, job):
        import pyre.util as util

        # Fix-up the job.
        if not job.task:
            # LSF scripts must have a job name; otherwise strange
            # "/bin/sh: Event not found" errors occur (tested on
            # TACC's Lonestar system).
            job.task = "jobname"
        job.walltime = util.hms(job.dwalltime.value)
        job.arguments = ' '.join(job.arguments)
        
        # Generate the main LSF batch script.
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
            from popen2 import Popen4

            cmd = [self.command]
            if self.wait:
                cmd.append("-K")
            self._info.log("spawning: %s" % ' '.join(cmd))
            child = Popen4(cmd)
            self._info.log("spawned process %d" % child.pid)

            print >> child.tochild, script
            child.tochild.close()

            if self.wait:
                self._info.log("Waiting for dispatch...")

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
        
        # "[When given the -K option], bsub will exit with the same
        # exit code as the job so that job scripts can take
        # appropriate actions based on the exit codes. bsub exits with
        # value 126 if the job was terminated while pending."
        if exitStatus == 0:
            pass
        elif self.wait:
            sys.exit(exitStatus)
        else:
            sys.exit("%s: %s: %s" % (sys.argv[0], cmd[0], statusStr))
        
        return


    def jobId(cls):
        return os.environ['LSB_JOBID']
    jobId = classmethod(jobId)


# end of file 
