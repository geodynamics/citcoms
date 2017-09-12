#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#<LicenseText>
#
# CitcomS.py by Eh Tan, Eun-seo Choi, and Pururav Thoutireddy.
# Copyright (C) 2002-2005, California Institute of Technology.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#
#</LicenseText>
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

"""Citcom input file parser.
(copied and modified from ConfigParser module in Python standard library)

Read and parse input file which consists of 'name=value' pairs. Note that
'name =value', `name= value', or 'name = value' is not allowed, i.e. space
is not allowed on either sides of the eqaul sign. 'name' is case-sensitive.
Blank lines, lines starting with '[', anything after `#', and just about
everything else is ignored.

Intrinsic defaults can be specified by passing them into the constructor as
a dictionary.


class:

Parser -- responsible for for parsing a list of
          input files, and managing the parsed database.

    methods:

    __init__(defaults=None)
        create the parser and specify a dictionary of intrinsic defaults.  The
        keys must be strings, the values must be appropriate for %()s string
        interpolation.

    has_option(option)
        return whether the given option exists.

    options()
        return list of configuration options.

    read(filenames)
        read and parse the list of named configuration files, given by
        name.  A single filename is also allowed.  Non-existing files
        are ignored.

    get(option)
        return a string value for the named option, based on the defaults
        passed into the constructor.

    getstr(option)
        return a string stripped of possible enclosing quotes.

    getint(options)
        like get(), but convert value to an integer.

    getfloat(options)
        like get(), but convert value to a float.

    getintvector(options)
        like get(), but convert value to an integer vector.

    getfloatvector(options)
        like get(), but convert value to a float vector.

    getboolean(options)
        like get(), but convert value to a boolean (currently case
        insensitively defined as 0, false, no, off for 0, and 1, true,
        yes, on for 1).  Returns 0 or 1.

"""
import os, sys, string, types

__all__ = ["NoOptionError", "ParsingError", "Parser"]



# exception classes
class Error(Exception):
    def __init__(self, msg=''):
        self._msg = msg
        Exception.__init__(self, msg)
    def __repr__(self):
        return self._msg
    __str__ = __repr__

class NoOptionError(Error):
    def __init__(self, option):
        Error.__init__(self, "No option `%s'" %
                       option)
        self.option = option

class ParsingError(Error):
    def __init__(self, filename):
        Error.__init__(self, 'File contains parsing errors: %s' % filename)
        self.filename = filename
        self.errors = []

    def append(self, lineno, line):
        self.errors.append((lineno, line))
        self._msg = self._msg + '\n\t[line %2d]: %s' % (lineno, line)


class Parser(object):
    def __init__(self,defaults=None):
        if defaults is None:
            self.__defaults = {}
        else:
            self.__defaults = defaults

    def defaults(self):
        return self.__defaults

    def options(self):
        """Return a list of option names."""
        return self.__options.keys()

    def has_option(self, option):
        """Check for the existence of a given option."""
        return self.__options.has_key(option)

    def read(self, filename):
        """Read and parse a filename."""
        fp = open(filename)
        self.__read(fp, filename)
        fp.close()

    def get(self, option):
        """Get an option value.
        """
        d = self.__defaults.copy()
        d.update(self.__options)
        # Update with the entry specific variables
        try:
            rawval = d[option]
        except KeyError:
            raise NoOptionError(option)

        return rawval

    def __get(self, conv, option):
        return conv(self.get(option))

    def __getvector(self, conv, option):
        val = []
        v = string.split(self.get(option),',')
        try:
            for item in v:
                val.append(conv(item))
            return val
        except TypeError, ValueError:
            raise ValueError, "`%s' is not a vector: %s" % (option, v)

    def getstr(self, option):
        v = self.get(option)
        # stripping quotation marks
        if v and v[0] == v[-1] and v[0] in '"\'':
            v = v[1:-1]
        return v

    def getint(self, option):
        return self.__get(int, option)

    def getintvector(self, option):
        return self.__getvector(int, option)

    def getfloat(self, option):
        return self.__get(float, option)

    def getfloatvector(self, option):
        return self.__getvector(float, option)

    def getboolean(self, option):
        states = {'1': 1, 'yes': 1, 'true': 1, 'on': 1,
                  '0': 0, 'no': 0, 'false': 0, 'off': 0}
        v = self.get(option)
        if not states.has_key(string.lower(v)):
            raise ValueError, 'Not a boolean: %s' % v
        return states[string.lower(v)]


    def __read(self, fp, fpname):
        """Parse a un-sectioned setup file.

        key/value options lines, indicated by `name=value' format lines.
        `name =value', `name= value', or `name = value' is not allowed.
        key is case-sensitive. Blank lines, anything after a `#',
        and just about everything else is ignored.
        """
        self.__options = {}
        lineno = 0
        e = None                                  # None, or an exception
        while 1:
            line = fp.readline()
            if not line:
                break
            lineno = lineno + 1
            # skip blank lines
            if string.strip(line) == '':
                continue
            # skip lines starting with '['
            if line[0] == '[':
                continue
            # remove anything after '#'
            line = string.split(line, '#')[0]
            # remove anything after ';'
            line = string.split(line, ';')[0]

            # key/value pairs can be seperated by whitespaces
            for opt in string.split(line):
                #if opt in string.whitespace:
                #    continue
                keyval = string.split(opt, '=')
                if len(keyval) == 2:
                    self.__options[keyval[0]] = keyval[1]
                else:
                    e = ParsingError(fpname)
                    e.append(lineno, `line`)

        # if any parsing errors occurred, raise an exception
        if e:
            raise e



if __name__ == '__main__':

    if len(sys.argv) != 2:
        import os.path
        print "usage: %s inputfile" % os.path.basename(sys.argv[0])
        sys.exit(1)

    parser = Parser()
    parser.read(sys.argv[1])

    #print parser.options()

    while 1:
        keyin = raw_input('key name: ')
        if keyin:
            for f in (parser.getint, parser.getfloat, parser.getboolean, parser.getstr, parser.getintvector, parser.getfloatvector):
                try:
                    val = None
                    if parser.has_option(keyin):
                        val = f(keyin)
                except (TypeError, ParsingError, ValueError):
                    continue
                print "%s=%s\n" % (keyin, str(val))

        else:
            break


# version
# $Id$

# End of file
