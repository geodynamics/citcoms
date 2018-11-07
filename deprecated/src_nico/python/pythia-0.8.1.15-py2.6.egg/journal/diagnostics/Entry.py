#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


class Entry(object):


    def attribute(self, key, value):
        self.meta[key] = value
        return


    def line(self, msg):
        self.text.append(msg)
        return


    def __init__(self):
        self.meta = {}
        self.text = []
        return


# version
__id__ = "$Id: Entry.py,v 1.1.1.1 2005/03/08 16:13:53 aivazis Exp $"

#  End of file 
