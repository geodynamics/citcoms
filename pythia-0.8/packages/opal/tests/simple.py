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
    print "<pre>"
    print "Hello!"

    import os
    import sys
    print sys.path

    for key, value in os.environ.iteritems():
        print "%s = {%s}" % (key,value)

    print "</pre>"
    print "</html>"

    return


# main
if __name__ == '__main__':
    test()


# version
__id__ = "$Id: simple.py,v 1.1.1.1 2005/03/15 06:09:10 aivazis Exp $"

# End of file 
