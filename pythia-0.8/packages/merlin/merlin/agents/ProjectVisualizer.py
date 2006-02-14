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
from pyre.weaver.components.Indenter import Indenter
from ProjectInspector import ProjectInspector


class ProjectVisualizer(Agent, ProjectInspector, Indenter):


    def execute(self, merlin, asset):
        if asset:
            print "\n".join(asset.identify(self))
        return True


    def onArchive(self, archive):
        text = [ self._render("archive '%s'" % archive.name) ]
        self._indent()
        text += archive.sources.identify(self)
        self._outdent()
        return text


    def onHeader(self, header):
        return self._render("header '%s'" % header.name)


    def onHeaders(self, headers):
        text = []
        for header in headers.assets:
            text.append(header.identify(self))
        return text


    def onLibrary(self, library):
        text = [ self._render("library '%s'" % library.name) ]
        self._indent()
        text += library.archive.identify(self)
        text += library.headers.identify(self)
        self._outdent()
        return text


    def onProject(self, project):
        text = [ self._render("project '%s'" % project.name) ]
        self._indent()
        for asset in project.assets:
            text += asset.identify(self)
        self._outdent()
        return text


    def onSource(self, source):
        return self._render("source '%s'" % source.name)


    def onSources(self, sources):
        text = []
        for source in sources.assets:
            text.append(source.identify(self))
        return text


    def __init__(self):
        Agent.__init__(self, 'visualizer', 'list')
        ProjectInspector.__init__(self)
        Indenter.__init__(self)
        return
            

# version
__id__ = "$Id: ProjectVisualizer.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
