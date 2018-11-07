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


def simple(tag):
    from SimpleLocator import SimpleLocator
    return SimpleLocator(tag)


def default():
    return simple('default')


def error():
    return simple('error')


def commandLine():
    return simple('command line')


def builtIn():
    return simple('built-in')


def script(source, line, function):
    from ScriptLocator import ScriptLocator
    return ScriptLocator(source, line, function)


def file(source, line=-1, column=-1):
    if line == -1 and column == -1:
        from SimpleFileLocator import SimpleFileLocator
        return SimpleFileLocator(source)
    
    from FileLocator import FileLocator
    return FileLocator(source, line, column)


def chain(this, next):
    from ChainLocator import ChainLocator
    return ChainLocator(this, next)


def stackTrace(st):
    from StackTraceLocator import StackTraceLocator
    return StackTraceLocator(st)


def here(depth=0):
    """return the current traceback as a locator"""
    import traceback
    from StackTraceLocator import StackTraceLocator
    st = traceback.extract_stack()
    depth += 1 # account for this function
    st = st[:-depth]
    return StackTraceLocator(st)


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:48 aivazis Exp $"

# End of file 
