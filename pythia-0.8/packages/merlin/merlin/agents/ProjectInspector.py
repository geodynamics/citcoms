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


class ProjectInspector(object):


    def onArchive(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onArchive'" % self.__class__.__name__)


    def onAsset(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onAsset'" % self.__class__.__name__)


    def onAssetContainer(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onAssetContainer'" % self.__class__.__name__)


    def onExecutable(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onExecutable'" % self.__class__.__name__)


    def onFileContainer(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onFileContainer'" % self.__class__.__name__)


    def onHeader(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onHeader'" % self.__class__.__name__)


    def onHeaders(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onHeaders'" % self.__class__.__name__)


    def onLibrary(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onLibrary'" % self.__class__.__name__)


    def onObject(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onObject'" % self.__class__.__name__)


    def onProject(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onProject'" % self.__class__.__name__)


    def onSharedObject(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onSharedObject'" % self.__class__.__name__)


    def onSource(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onSource'" % self.__class__.__name__)


    def onSources(self, asset):
        raise NotImplementedError(
            "class '%s' must override 'onSources'" % self.__class__.__name__)


# version
__id__ = "$Id: ProjectInspector.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
