#------------------------------------------------------------------------------
# Copyright 2003, Enthought, Inc.
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# 
# Author: Enthought, Inc.
# Description: <Enthought library component>
#------------------------------------------------------------------------------
"""
==========================
Enthought Library
========================== 

The Enthought Open Source Library

Copyright 2003-2005, Enthought, Inc.


"""

try:
    import __config__
except ImportError:
    __config__ = None

if __config__ is not None:
    if __config__.get_info('numpy'):
        import os
        if os.environ.get('NUMERIX','numpy').lower() not in ['','numpy']:
            print 55*'*'
            print "*** This enthought installation is built against numpy\n"\
                  "*** but the current environment is for %r.\n"\
                  "*** Resetting NUMERIX variable to 'numpy'."% (os.environ['NUMERIX'])
            print 55*'*'
        os.environ['NUMERIX'] = 'numpy'
