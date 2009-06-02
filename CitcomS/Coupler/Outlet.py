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



class Outlet(object):

    def __init__(self):
        self._handle = None
        return


    def send(self):
        import ExchangerLib
        ExchangerLib.Outlet_send(self._handle)
        return





class SVTOutlet(Outlet):

    def __init__(self, source, all_variables):
        import ExchangerLib
        self._handle = ExchangerLib.SVTOutlet_create(source,
                                                     all_variables)
        return




class TOutlet(Outlet):

    def __init__(self, source, all_variables):
        import ExchangerLib
        self._handle = ExchangerLib.TOutlet_create(source,
                                                   all_variables)
        return




class POutlet(Outlet):

    def __init__(self, source, all_variables):
        import ExchangerLib
        self._handle = ExchangerLib.POutlet_create(source,
                                                   all_variables)
        return




class VTOutlet(Outlet):

    def __init__(self, source, all_variables):
        import ExchangerLib
        self._handle = ExchangerLib.VTOutlet_create(source,
                                                    all_variables)
        return


class VOutlet(Outlet):

    def __init__(self, source, all_variables):
        import ExchangerLib
        self._handle = ExchangerLib.VOutlet_create(source,
                                                   all_variables)
        return


"""
class TractionOutlet(Outlet):


    def __init__(self, source, all_variables, mode='F'):
        import ExchangerLib
        self._handle = ExchangerLib.TractionOutlet_create(source,
                                                          all_variables,
                                                          mode)
        return

"""


# version
__id__ = "$Id$"

# End of file
