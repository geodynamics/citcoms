#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



class Inlet(object):

    def __init__(self):
        self._handle = None
        return


    def impose(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_impose(self._handle)
        return


    def recv(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_recv(self._handle)
        return


    def storeTimestep(self, fge_t, cge_t):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_storeTimestep(self._handle, fge_t, cge_t)
        return




class SVTInlet(Inlet):

    def __init__(self, mesh, sink, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.SVTInlet_create(mesh,
                                                 sink,
                                                 all_variables)
        return




class TInlet(Inlet):

    def __init__(self, mesh, sink, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TInlet_create(mesh,
                                               sink,
                                               all_variables)
        return




class VTInlet(Inlet):

    def __init__(self, mesh, sink, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.VTInlet_create(mesh,
                                                sink,
                                                all_variables)
        return



"""
class BoundaryVTInlet(Inlet):
    '''Available modes -- see above
    '''


    def __init__(self, communicator, boundary, sink, all_variables, mode="VT"):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.BoundaryVTInlet_create(communicator.handle(),
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
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TractionInlet_create(boundary,
                                                      sink,
                                                      all_variables,
                                                      mode)
        return

"""

# version
__id__ = "$Id: Inlet.py,v 1.6 2004/05/11 07:59:31 tan2 Exp $"

# End of file
