#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


def test():
    print "Content-type: text/html"
    print
    print "<html>"
    print "<head></head>"
    print "<body>"

    import os
    import sys
    print "python sys.path:", sys.path
    print ""

    print "<h1>ENVIRONMENT:</h1>"
    print "<pre>"
    for key, value in os.environ.iteritems():
        print "%s = {%s}" % (key,value)

    print "</pre>"
    print "</body>"
    print "</html>"

    return


# main
if __name__ == '__main__':
    test()


# version
__id__ = "$Id: simple.py,v 1.1 2005/03/16 22:32:02 aivazis Exp $"

# End of file 
