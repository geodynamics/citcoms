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

    #import journal
    #journal.info("staging").activate()
    #journal.debug("staging").activate()
    #
    #app = FullApp("full")
    #app.main()

    import os, sys
    # basic arguments for a full citcoms run
    cmd = './citcomsregional.py --solver=full --param.datafile=fulltest --staging.nodegen="n%03d" --staging.nodelist=[131-168] --staging.nodes=12'

    # append other command line arguments
    for arg in sys.argv[1:]:
        cmd = cmd + " " + arg

    print cmd
    os.system(cmd)


# version
__id__ = "$Id: citcomsfull.py,v 1.4 2003/09/12 16:25:49 tan2 Exp $"

#  End of file
