#!/usr/bin/env python
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
    journal.debug("staging").activate()

    app = RegionalApp()
    app.main()


# version
__id__ = "$Id: citcomsregional.py,v 1.2 2003/04/10 23:40:55 tan2 Exp $"

#  End of file 
