#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2007  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from pyre.applications import AppRunner
from pyre.util import loadObject


class BatchScript(AppRunner):


    import pyre.inventory as pyre
    schedulerClass = pyre.str("scheduler-class", default="pyre.schedulers:SchedulerNone")


    def _init(self):
        super(BatchScript, self)._init()
        self.SchedulerClass = loadObject(self.schedulerClass)
        return


    def defineMacros(self, macros):
        macros['job.id'] = self.SchedulerClass.jobId()
        return


    def runSubscript(self, *args, **kwds):
        macros = kwds.setdefault('macros', {})
        self.defineMacros(macros)
        super(BatchScript, self).runSubscript(*args, **kwds)
        return


# end of file 
