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



class Inlet(object):

    def __init__(self):
        self._handle = None
        return


    def impose(self):
        import ExchangerLib
        ExchangerLib.Inlet_impose(self._handle)
        return


    def recv(self):
        import ExchangerLib
        ExchangerLib.Inlet_recv(self._handle)
        return


    def storeTimestep(self, fge_t, cge_t):
        import ExchangerLib
        ExchangerLib.Inlet_storeTimestep(self._handle, fge_t, cge_t)
        return




class SVTInlet(Inlet):

    def __init__(self, mesh, sink, all_variables):
        import ExchangerLib
        self._handle = ExchangerLib.SVTInlet_create(mesh,
                                                    sink,
                                                    all_variables)
        return



class BoundarySVTInlet(Inlet):

    def __init__(self, mesh, sink, all_variables, comm):
        import ExchangerLib
        self._handle = ExchangerLib.BoundarySVTInlet_create(mesh,
                                                            sink,
                                                            all_variables,
                                                            comm)
        return



class TInlet(Inlet):

        def __init__(self, mesh, sink, all_variables):
                import ExchangerLib
                self._handle = ExchangerLib.TInlet_create(mesh,
                                                          sink,
                                                          all_variables)
                return


class PInlet(Inlet):

        def __init__(self, mesh, sink, all_variables):
                import ExchangerLib
                self._handle = ExchangerLib.PInlet_create(mesh,
                                                          sink,
                                                          all_variables)
                return


class SInlet(Inlet):

        def __init__(self, mesh, sink, all_variables):
                import ExchangerLib
                self._handle = ExchangerLib.SInlet_create(mesh,
                                                          sink,
                                                          all_variables)
                return


class VTInlet(Inlet):

        def __init__(self, mesh, sink, all_variables):
                import ExchangerLib
                self._handle = ExchangerLib.VTInlet_create(mesh,
                                                           sink,
                                                           all_variables)
                return



"""
class BoundaryVTInlet(Inlet):
    '''Available modes -- see above
    '''


    def __init__(self, communicator, boundary, sink, all_variables, mode="VT"):
        import ExchangerLib
        self._handle = ExchangerLib.BoundaryVTInlet_create(communicator.handle(),
                                                           boundary,
                                                           sink,
                                                           all_variables,
                                                           mode)
        return




class TractionInlet(Inlet):
    '''Inlet that impose velocity and/or traction on the boundary
    Available modes --
    "F": traction only
    "V": velocity only
    "FV": normal velocity and tangent traction
    '''

    def __init__(self, boundary, sink, all_variables, mode='F'):
        import ExchangerLib
        self._handle = ExchangerLib.TractionInlet_create(boundary,
                                                         sink,
                                                         all_variables,
                                                         mode)
        return

"""

# version
__id__ = "$Id: Inlet.py 15108 2009-06-02 22:56:46Z tan2 $"

# End of file
