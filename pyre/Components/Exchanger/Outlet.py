#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# <LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#



class Outlet(object):

    def __init__(self):
        self._handle = None
        return


    def send(self):
        raise NotImplementedError
        return





class VTOutlet(Outlet):
    '''Available modes --
    "V": send velocity
    "T": send temperature
    "VT": send both velocity and temperature
    '''

    def __init__(self, source, all_variables, mode="VT"):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.VTOutlet_create(source,
                                                 all_variables,
                                                 mode)
        return


    def send(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.VTOutlet_send(self._handle)
        return




class TractionOutlet(Outlet):


    def __init__(self, source, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TractionOutlet_create(source,
                                                       all_variables)
        return


    def send(self):
        import CitcomS.Exchanger as Exchanger
        Exchanger.TractionOutlet_send(self._handle)
        return



# version
__id__ = "$Id: Outlet.py,v 1.1 2004/02/26 22:29:30 tan2 Exp $"

# End of file
