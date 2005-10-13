#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             Michael A.G. Aivazis
#                      California Institute of Technology
#                      (C) 1998-2005  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

print """

pyre.geometry:
  - single vs. multi body in the interfaces modellers, parsers and renderers
  - sort out the intrinsic length scale for models
  - models as databases for named parts


pyre.weaver
  - rendering a document fragment into an existing stream (e.g. do not generate headers/footers)


pyre.xml:
  - namespaces
  - recognizing embedded documents and  dispatching them to the correct parser


pyre.services:
  - implement KEEPALIVE for TCPSessions so that I don't have to make a new connection
    for every request. This will likely require reimplementation of the way a Service
    interacts with its selector, so that it gets a chance to register a new callback


pyre.ipa:
  - break it up into two services: a password database manager and a ticket service


"""

# version
__id__ = "$Id: TODO.py,v 1.1.1.1 2005/03/08 16:13:28 aivazis Exp $"

# End of file 
