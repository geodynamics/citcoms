#!/usr/bin/env mpipython.exe
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  <LicenseText>
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from CitcomS.CoupledRegionalApp import CoupledRegionalApp


# main
if __name__ == "__main__":

    import journal
    journal.info("staging").activate()
    journal.debug("staging").activate()

    app1 = CoupledRegionalApp("r1")
    app1.inventory.ranklist = [0,1,2,3]
    app1.main()

    import sys
    sys.exit(0)



    app2 = RegionalApp("r2")


    #app1.main()

    #app2.main()

# version
__id__ = "$Id: coupledcitcoms.py,v 1.1 2003/08/22 22:19:48 tan2 Exp $"

#  End of file
