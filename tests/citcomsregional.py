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
    app = RegionalApp(sys.argv[1])
    app.main()


# version
__id__ = "$Id: citcomsregional.py,v 1.4 2003/06/13 17:11:32 tan2 Exp $"

#  End of file 
