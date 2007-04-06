#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from mpi.Application import Application
import journal


class BaseApplication(Application):


    def __init__(self, name="CitcomS"):
        Application.__init__(self, name)

        self._info = journal.debug("application")
        return



    def _init(self):
        Application._init(self)
        self.nodes = self.getNodes()
        return



    def main(self, *args, **kwds):
        self.initialize()
        self.reportConfiguration()
        self.launch()
        return



    def launch(self):
        self.controller.launch(self)

        self.controller.march(steps=self.inventory.steps)
        return



    def _getPrivateDepositoryLocations(self):
        from os.path import dirname, isdir, join
        list = []
        etc = join(dirname(dirname(__file__)), 'etc')
        if isdir(etc):
            # The user is running directly from the source directory.
            list.append(etc)
        else:
            try:
                from config import makefile
                pkgsysconfdir = makefile['pkgsysconfdir']
                if isdir(pkgsysconfdir):
                    list.append(pkgsysconfdir)
            except ImportError, KeyError:
                pass
        return list




    class Inventory(Application.Inventory):

        import pyre.inventory

        steps = pyre.inventory.int("steps", default=1)



# version
__id__ = "$Id$"

# End of file
