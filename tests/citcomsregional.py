#!/usr/bin/env mpipython.exe
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#  <LicenseText>
# 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from CitcomS.RegionalApp import RegionalApp


# main
if __name__ == "__main__":

    import journal
    journal.info("staging").activate()
    journal.debug("staging").activate()

    import sys
    app = RegionalApp()
    app.main()


# version
__id__ = "$Id: citcomsregional.py,v 1.5 2003/07/24 17:46:47 tan2 Exp $"

#  End of file 
