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


from merlin.assets.Project import Project


class Package(Project):


    def package(self, name=None):
        if name is None:
            name = self.name

        from merlin.assets.PythonPackage import PythonPackage
        package = PythonPackage(name)

        self.assets.append(package)

        return package


    def __init__(self, name):
        Project.__init__(self, name)
        return


# version
__id__ = "$Id: Package.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

# End of file 
