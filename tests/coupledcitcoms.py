#!/usr/bin/env mpipython.exe
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from CitcomS.CoupledApp import CoupledApp


# main
if __name__ == "__main__":

    import journal
    #journal.debug("Array2D").activate()
    #journal.debug("initTemperature").activate()
    #journal.debug("imposeBC").activate()
    journal.debug("Exchanger").activate()
    journal.info("  X").activate()
    journal.info("  proc").activate()
    journal.info("  bid").activate()

    app = CoupledApp("app")
    app.main()


# version
__id__ = "$Id: coupledcitcoms.py,v 1.3 2003/10/24 05:23:36 tan2 Exp $"

#  End of file
