#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from AssetContainer import AssetContainer


class PythonPackage(AssetContainer):


    def identify(self, inspector):
        return inspector.onPythonPackage(self)


    def __init__(self, name):
        AssetContainer.__init__(self, name)

        from Sources import Sources
        self._modules = Sources(self.name)

        return


    def _getModules(self):
        return self._modules


    def _setModules(self, modules):
        from Source import Source
        sources = [ Source(name) for name in modules ]
        self._modules.append(sources)
        return


    modules = property(_getModules, _setModules, None, "")

# version
__id__ = "$Id: PythonPackage.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
