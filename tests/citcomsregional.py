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


from CitcomS.CitcomSRegionalApp import CitcomSRegionalApp


# main
if __name__ == "__main__":

    import journal
    #journal.debug("mesher").activate()
    #journal.debug("mesher.phases").activate()
    journal.debug("staging").activate()

    app = CitcomSRegionalApp()
    app.main()


# version
__id__ = "$Id: citcomsregional.py,v 1.1 2003/03/26 00:49:05 ces74 Exp $"

#  End of file 
