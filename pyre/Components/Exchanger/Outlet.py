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




class TOutlet(Outlet):

    def __init__(self, source, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TOutlet_create(source,
                                                all_variables)
        return




class VTOutlet(Outlet):

    def __init__(self, source, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.VTOutlet_create(source,
                                                 all_variables)
        return


class VOutlet(Outlet):

    def __init__(self, source, all_variables):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.VOutlet_create(source,
                                                all_variables)
        return


"""
class TractionOutlet(Outlet):


    def __init__(self, source, all_variables, mode='F'):
        import CitcomS.Exchanger as Exchanger
        self._handle = Exchanger.TractionOutlet_create(source,
                                                       all_variables,
                                                       mode)
        return

"""


# version
__id__ = "$Id: Outlet.py,v 1.5 2004/05/18 21:15:26 ces74 Exp $"

# End of file
