#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                              Michael A.G. Aivazis
#                       California Institute of Technology
#                       (C) 1998-2005  All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


def project(name, mode):
    from Project import Project
    return Project(name, mode)


def library(name):
    from Library import Library
    return Library(name)


def archive(name):
    from Archive import Archive
    return Archive(name)


def sources(name):
    from Sources import Sources
    return Sources(name)


def headers(name):
    from Headers import Headers
    return Headers(name)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:59 aivazis Exp $"

#  End of file 
