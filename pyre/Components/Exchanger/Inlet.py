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
        raise NotImplementedError
        return


    def recv(self):
        raise NotImplementedError
        return


    def storeTimestep(self, fge_t, cge_t):
        import CitcomS.Exchanger as Exchanger
        Exchanger.Inlet_storeTimestep(self._handle, fge_t, cge_t)
        return




class VTInlet(Inlet):
    '''Available modes --
    "V": impose velocity as BC
    "T": impose temperature as BC
    "t": impose temperature but not as BC
    "VT": "V" + "T"

    mode "T" and mode "t" cannot co-exist
    '''

    def __init__(self, mesh, sink, all_variables, mode="VT"):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.VTInlet_create(mesh,
                                                sink,
                                                all_variables,
                                                mode)
        return


    def impose(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.VTInlet_impose(self._handle)
        return


    def recv(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.VTInlet_recv(self._handle)
        return




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


    def impose(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.BoundaryVTInlet_impose(self._handle)
        return


    def recv(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.BoundaryVTInlet_recv(self._handle)
        return




# version
__id__ = "$Id: Inlet.py,v 1.2 2004/03/11 01:07:19 tan2 Exp $"

# End of file
