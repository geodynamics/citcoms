#!/usr/bin/env mpipython.exe
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from CitcomS.FullApp import FullApp


# main
if __name__ == "__main__":

    import journal
    journal.info("staging").activate()
    journal.debug("staging").activate()

    app = FullApp("full")
    app.main()


# version
__id__ = "$Id: citcomsfull.py,v 1.3 2003/08/22 22:19:48 tan2 Exp $"

#  End of file
