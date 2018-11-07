#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                      California Institute of Technology
#                        (C) 2009  All Rights Reserved
#
# {LicenseText}
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class Parser(object):


    def __init__(self, root):
        self.root = root
        self.lexer = None
        self.nextToken = None
        self.acceptedToken = None
        return


    def parseFile(self, pathname):
        self.pathname = pathname
        stream = open(pathname, "r")
        self.lexer = self._createLexer(stream.read())
        self._parse()
        return


    def _parse(self):
        self._getToken()
        while not self._accept("eof"):
            name = self._parseName()
            self._expect("leftBrace")
            childNode = self._getNode(self.root, name)
            self._parseTraitSet(childNode)
        return


    def _parseTraitSet(self, node):
        while not self._accept("rightBrace"):
            name = self._parseName()
            if self._accept("colon"):
                self._parseTrait(node, name)
            elif self._accept("leftBrace"):
                childNode = self._getNode(node, name)
                self._parseTraitSet(childNode)
            else:
                self._parseError()
        return


    def _parseTrait(self, node, name):
        if (self._accept("string") or
            self._accept("word")):
            pass
        else:
            self._parseError()
        value = self.acceptedToken.value
        locator = self._getLocator()
        self._expect("semicolon")
        key = name[-1]
        path = name[:-1]
        node = self._getNode(node, path)
        node.setProperty(key, value, locator)
        return


    def _parseName(self):
        self._expect("word")
        name = self.acceptedToken.value
        while self._accept("word"):
            name += self.acceptedToken.value
        return name.split('.')


    def _expect(self, kind):
        if self._accept(kind):
            return
        self._parseError()


    def _accept(self, kind):
        if self.nextToken.kind == kind:
            self._getToken()
            return True
        return False


    def _getToken(self):
        self.acceptedToken = self.nextToken
        self.nextToken = self.lexer.nextToken()
        return


    def _getNode(self, node, path):
        if len(path) == 0:
            return node
        key = path[0]
        return self._getNode(node.getNode(key), path[1:])


    def _getLocator(self):
        import pyre.parsing.locators as locators
        return locators.file(self.pathname, self.lexer.lineNo)


    def _createLexer(self, input):
        from Lexer import Lexer
        return Lexer(input)


    def _parseError(self):
        raise self.ParseError("line %d: '%s' unexpected" %
                              (self.lexer.lineNo, self.nextToken.value))

    class ParseError(Exception):
        pass



# end of file
