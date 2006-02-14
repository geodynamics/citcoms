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


from Agent import Agent
from ProjectInspector import ProjectInspector


class Janitor(Agent, ProjectInspector):


    def execute(self, merlin, asset):
        if asset:
            asset.identify(self)

        # tidy up
        self._tidy()
        return True


    def onLibrary(self, library):
        print "    %s: library '%s'" % (self.mode, library.name)
        return True


    def onProject(self, project):
        print "%s: project '%s'" % (self.mode, project.name)

        status = True

        for asset in project.assets:
            status = asset.identify(self)
            if not status:
                break
            
        return status


    def __init__(self, name=None, project=None, mode=None):
        if name is None:
            name = "janitor"

        if mode is None:
            mode = 'clean'
        self.mode = mode

        self.project = project
            
        Agent.__init__(self, name, "janitor")
        ProjectInspector.__init__(self)

        return


    def _tidy(self):
        import os
        os.system("rm -f *~ .*~ core *.bak *.pyc")
        return
    

# version
__id__ = "$Id: Janitor.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
