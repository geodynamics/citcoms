#!/usr/bin/env mpipython.exe
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from CitcomS.SimpleApp import SimpleApp
#from CitcomS.RegionalApp import RegionalApp


# main
if __name__ == "__main__":

    import journal
    journal.info("staging").activate()
    journal.debug("staging").activate()
    journal.debug("application").activate()

    app = SimpleApp("regional")

    #print app.name
    print dir(app.inventory)
    #print app.inventory.controller.name
    #print dir(app.inventory.controller.inventory)

    app.main()


# version
__id__ = "$Id: citcomsregional.py,v 1.9 2003/09/12 16:25:49 tan2 Exp $"

#  End of file
