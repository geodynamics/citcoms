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


class Lexer(object):


    def __init__(self, input):
        import re
        self.regexp = re.compile(r""
                                 "(?P<leftBrace>{)|"
                                 "(?P<rightBrace>})|"
                                 "(?P<colon>:)|"
                                 "(?P<semicolon>;)|"
                                 "\"(?P<string>[^\"\n]*)\"|"
                                 "(?P<newline>\n)|"
                                 "(?P<word>[^\s{}:;]+)")
        input = self._stripComments(input)
        self.tokenIter = self._tokenIter(input)
        self.lineNo = 1
        return


    def nextToken(self):
        try:
            token = self.tokenIter.next()
            while token.kind == "newline":
                self.lineNo += 1
                token = self.tokenIter.next()
        except StopIteration:
            token = self.Token("eof", None)
        return token


    def _stripComments(self, input):
        import re

        # strip single-line comments
        regexp = re.compile(r"//[^\n]*")
        input = ''.join(filter(None, regexp.split(input)))
        
        # strip multi-line comments
        regexp = re.compile(r"(/\*)|(\*/)")
        newlines = re.compile(r"[\n\r]")
        result = ""
        inComment = False
        for chunk in filter(None, regexp.split(input)):
            if chunk == "/*":
                assert not inComment, "'/*' inside of comment"
                inComment = True
            elif chunk == "*/":
                assert inComment, "'*/' outside of comment"
                inComment = False
            elif inComment:
                # preserve newlines for the sake of line numbering
                result += ''.join(newlines.findall(chunk))
            else:
                result += chunk

        return result
            

    def _tokenIter(self, input):
        for match in self.regexp.finditer(input):
            yield self.Token(match.lastgroup, match.group(match.lastindex))
        return


    class Token(object):

        __slots__ = ('kind', 'value')

        def __init__(self, kind, value):
            self.kind = kind
            self.value = value
            return


# end of file
