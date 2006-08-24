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



    def main(self, *args, **kwds):
        self.initialize()
        self.reportConfiguration()
        self.launch()
        return



    def launch(self):
        self.controller.launch(self)

        self.controller.march(steps=self.inventory.steps)
        return



    def usage(self):
        print 'usage: citcoms [<property>=<value>] [<facility>.<property>=<value>] ...'
        self.showUsage()
        print """\
For more information about a particular component:
  --<facility>.help-properties
  --<facility>.help-components
where <facility> is the facility to which the component is bound; e.g.:
  %(name)s --launcher.help-properties""" % locals()
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



    def initializeCurator(self, curator, registry):
        from Components.CodecConfig import CodecConfig
        cfg = CodecConfig()
        curator.registerCodecs(cfg)
        return super(BaseApplication, self).initializeCurator(curator, registry)



    def collectUserInput(self, registry):
        # read INI-style .cfg files
        import journal
        error = journal.error(self.name)
        from Components.CodecConfig import CodecConfig
        curator = self.getCurator()
        configRegistry = curator.getTraits(self.name, extraDepositories=[], encoding='cfg')
        self.updateConfiguration(configRegistry)
        # read parameter files given on the command line
        from os.path import isfile, splitext
        for arg in self.argv:
            if isfile(arg):
                base, ext = splitext(arg)
                encoding = ext[1:] # NYI: not quite
                codec = self.getCurator().codecs.get(encoding)
                if codec:
                    shelf = codec.open(base)
                    paramRegistry = shelf['inventory'].getFacility(self.name)
                    if paramRegistry:
                        self.updateConfiguration(paramRegistry)
                else:
                    error.log("unknown encoding: %s" % ext)
            else:
                error.log("cannot open '%s'" % arg)
        return




    class Inventory(Application.Inventory):

        import pyre.inventory

        import Controller
        import Solver

        launcher = pyre.inventory.facility("launcher", default="mpich")

        steps = pyre.inventory.int("steps", default=1)



# version
__id__ = "$Id$"

# End of file
