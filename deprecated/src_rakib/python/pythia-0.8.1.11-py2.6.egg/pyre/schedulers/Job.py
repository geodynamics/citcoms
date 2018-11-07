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


from pyre.components import Component


class Job(Component):
    
    
    name = "job"
    
    
    import pyre.inventory as pyre
    import pyre.util as util
    from pyre.units.time import minute

    task              = pyre.str("name") # 'task' internally, to avoid name conflict
    
    queue             = pyre.str("queue")
    mail              = pyre.bool("mail", default=False)
    dwalltime         = pyre.dimensional("walltime", default=0*minute)
    
    stdin             = pyre.str("stdin", default=util.devnull())
    stdout            = pyre.str("stdout", default="stdout.txt")
    stderr            = pyre.str("stderr", default="stderr.txt")

    environment       = pyre.list("environment")

    executable        = pyre.str("executable")
    arguments         = pyre.list("arguments")

    comments          = pyre.list("comments")

    
    def __init__(self):
        super(Job, self).__init__()
        self.nodes = 1
        self.id = None


    def getStateArgs(self, stage):
        state = []
        if stage == 'launch':
            state.append("--macros.job.name=%s" % self.task)
        elif stage == 'compute':
            state.append("--macros.job.id=%s" % self.id)
        return state


# end of file
