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
    app.main()


# version
__id__ = "$Id: citcomsfull.py,v 1.1 2003/08/01 22:57:42 tan2 Exp $"

#  End of file 
