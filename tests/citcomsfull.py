#!/usr/bin/env mpipython.exe
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from CitcomS.FullApp import FullApp as RegionalApp


# main
if __name__ == "__main__":

    import journal
    journal.info("staging").activate()
    journal.debug("staging").activate()

    import sys
    app = RegionalApp()
    app.inventory.staging.inventory.nodes = \
                                  app.inventory.mesher.inventory.nprocx * \
                                  app.inventory.mesher.inventory.nprocy * \
                                  app.inventory.mesher.inventory.nprocz * 12
    app.main()


# version
__id__ = "$Id: citcomsfull.py,v 1.2 2003/08/20 02:45:23 tan2 Exp $"

#  End of file
