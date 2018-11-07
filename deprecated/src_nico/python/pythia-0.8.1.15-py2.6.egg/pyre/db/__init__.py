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


def connect(database, wrapper=None):

    if wrapper is None or wrapper == "psycopg":
        from Psycopg import Psycopg
        return Psycopg(database)

    import journal
    journal.error("pyre.db").log("%r: unknown db wrapper type" % wrapper)
    return None


def bigint(**kwds):
    from BigInt import BigInt
    return BigInt(**kwds)


def boolean(**kwds):
    from Boolean import Boolean
    return Boolean(**kwds)


def char(**kwds):
    from Char import Char
    return Char(**kwds)


def date(**kwds):
    from Date import Date
    return Date(**kwds)


def double(**kwds):
    from Double import Double
    return Double(**kwds)


def integer(**kwds):
    from Integer import Integer
    return Integer(**kwds)


def interval(**kwds):
    from Interval import Interval
    return Interval(**kwds)


def real(**kwds):
    from Real import Real
    return Real(**kwds)


def smallint(**kwds):
    from SmallInt import SmallInt
    return SmallInt(**kwds)


def time(**kwds):
    from Time import Time
    return Time(**kwds)


def timestamp(**kwds):
    from Timestamp import Timestamp
    return Timestamp(**kwds)


def varchar(**kwds):
    from VarChar import VarChar
    return VarChar(**kwds)


# version
__id__ = "$Id: __init__.py,v 1.4 2005/04/06 21:02:28 aivazis Exp $"

# End of file 
