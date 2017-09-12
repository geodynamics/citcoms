#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2006  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


from ConfigParser import SafeConfigParser
import pyre.parsing.locators as locators
from pyre.util import expandMacros


class Parser(SafeConfigParser):

    """Python's SafeConfigParser, hacked to provide file & line info,
    and extended to populate a Pyre Registry."""

    # This class goes to extreme lengths to extract file and line info
    # from Python's ConfigParser module.  It injects proxy 'file' and
    # 'dict' objects into ConfigParser's code.  As the file is parsed,
    # the proxy objects gain control, spying upon the parsing process.
    
    # We should probably write our own parser... *sigh*

    class FileProxy(object):

        def __init__(self):
            self.fp = None
            self.name = "unknown"
            self.lineno = 0

        def readline(self, size=-1):
            line = self.fp.readline()
            if line:
                self.lineno = self.lineno + 1
            return line

        def __getattr__(self, name):
            return getattr(self.fp, name)

    class Section(dict):

        def __init__(self, sectname, node, macros, fp):
            dict.__init__(self)
            self.node = node
            self.macros = macros
            self.fp = fp
        
        def __setitem__(self, trait, value):
            locator = locators.file(self.fp.name, self.fp.lineno)
            path = trait.split('.')
            key = path[-1]
            path = path[:-1]
            node = _getNode(self.node, path)
            value = expandMacros(value, self.macros)
            node.setProperty(key, value, locator)
            dict.__setitem__(self, key, value)

    class SectionDict(dict):

        def __init__(self, root, macros):
            dict.__init__(self)
            self.root = root
            self.macros = macros
            self.fp = Parser.FileProxy()
        
        def __contains__(self, sectname):
            # Prevent 'ConfigParser' from creating section
            # dictionaries; instead, create our own.
            if not dict.__contains__(self, sectname):
                node = _getNode(self.root, sectname.split('.'))
                cursect = Parser.Section(sectname, node, self.macros, self.fp)
                self[sectname] = cursect
            return True
        
        def __setitem__(self, key, value):
            dict.__setitem__(self, key, value)

    def __init__(self, root, defaults=None, macros=None):
        SafeConfigParser.__init__(self, defaults)
        if macros is None:
            macros = dict()
        self._sections = Parser.SectionDict(root, macros)

    def _read(self, fp, fpname):
        self._sections.fp.fp = fp
        self._sections.fp.name = fpname
        self._sections.fp.lineno = 0
        SafeConfigParser._read(self, self._sections.fp, fpname)
        self._sections.fp.__init__()

    def optionxform(self, optionstr):
        # Don't lower() option names.
        return optionstr


def _getNode(node, path):
    if len(path) == 0:
        return node
    key = path[0].strip()
    return _getNode(node.getNode(key), path[1:])


# end of file
