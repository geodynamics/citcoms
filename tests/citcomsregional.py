#!/usr/bin/env mpipython.exe
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2003 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


from CitcomS.RegionalApp import RegionalApp


# main
if __name__ == "__main__":

    import journal
    #journal.debug("mesher").activate()
    #journal.debug("mesher.phases").activate()
    journal.info("staging").activate()
    journal.debug("staging").activate()

    import sys
    app = RegionalApp(sys.argv[1])
    app.main()


# version
__id__ = "$Id: citcomsregional.py,v 1.3 2003/05/16 21:11:54 tan2 Exp $"

#  End of file 
