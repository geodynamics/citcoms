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


import urllib


class CGIParser(object):


    def parse(self, registry, arg, source):
        import pyre.parsing.locators
        self.locator = pyre.parsing.locators.simple(source)

        for field in arg.split(self.argsep):
            tokens = field.split(self.assignment)
            try:
                key, value = tokens
            except ValueError:
                if self.strict:
                    raise ValueError, "bad query field {%r}" % field
                elif self.keepBlanks:
                    key = field
                    value = ''
                else:
                    continue

            self._processArgument(key, value, registry)

        self.locator = None

        return


    def __init__(self, strict=False, keepBlanks=False):
        self.argsep = '&'
        self.fieldsep = '.'
        self.assignment = '='

        self.strict = strict
        self.keepBlanks = keepBlanks

        # options tracer
        self.locator = None

        return


    def _processArgument(self, key, value, registry):

        fields = key.split(self.fieldsep)

        children = []
        for level, field in enumerate(fields):
            if field[0] == '[' and field[-1] == ']':
                candidates = field[1:-1].split(',')
            else:
                candidates = [field]
            children.append(candidates)

        self._storeValue(registry, children, value)

        return


    def _storeValue(self, node, children, value):
        if len(children) == 1:
            for key in children[0]:
                key = key.strip()
                node.setProperty(key, urllib.unquote(value), self.locator)
            return

        for key in children[0]:
            self._storeValue(node.getNode(key), children[1:], urllib.unquote(value))

        return


# version
__id__ = "$Id: CGIParser.py,v 1.2 2005/05/03 03:04:21 pyre Exp $"

#  End of file 
