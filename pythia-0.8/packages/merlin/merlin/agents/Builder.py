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


class Builder(Agent, ProjectInspector):


    def onLibrary(self, library):
        print "    building library '%s'" % library.name
        return True


    def onProject(self, project):
        print "building project '%s'" % project.name

        status = True

        for asset in project.assets:
            status = asset.identify(self)
            if not status:
                break
            
        return status


    def __init__(self, name=None):
        if name is None:
            name = "builder"
            
        Agent.__init__(self, name, "build")
        ProjectInspector.__init__(self)

        return
    

# version
__id__ = "$Id: Builder.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
