#!/usr/bin/env python
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#                               Michael A.G. Aivazis
#                        California Institute of Technology
#                        (C) 1998-2005 All Rights Reserved
# 
#  <LicenseText>
# 
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


def tmp():
    from tmpdir import tmp
    return tmp()


def devnull():
    import os.path
    try:
        return os.path.devnull
    except AttributeError:
        return "/dev/null"


def hms(t):
    return (int(t / 3600), int((t % 3600) / 60), int(t % 60))


def spawn(onParent, onChild):
    from subprocesses import spawn
    return spawn(onParent, onChild)


def spawn_pty(onParent, onChild):
    from subprocesses import spawn_pty
    return spawn_pty(onParent, onChild)


def expandMacros(raw, substitutions):
    from expand import expandMacros
    return expandMacros(raw, substitutions)


def loadObject(name):
    """Load and return the object referenced by <name> ==
    some.module[:some.attr].
    """

    module, attrs = name.split(':')
    attrs = attrs.split('.')
    obj = __import__(module, globals(), globals(), ['__name__'])
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


# version
__id__ = "$Id: __init__.py,v 1.1.1.1 2005/03/08 16:13:41 aivazis Exp $"

#  End of file 
