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
        import CitcomS.Exchanger as Exchanger
        Exchanger.Outlet_send(self._handle)
        return





class SVTOutlet(Outlet):

    def __init__(self, source, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.SVTOutlet_create(source,
                                                  all_variables)
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




class TractionOutlet(Outlet):


    def __init__(self, source, all_variables, mode='F'):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TractionOutlet_create(source,
                                                       all_variables,
                                                       mode)
        return




# version
__id__ = "$Id: Outlet.py,v 1.3 2004/04/16 00:05:50 tan2 Exp $"

# End of file
