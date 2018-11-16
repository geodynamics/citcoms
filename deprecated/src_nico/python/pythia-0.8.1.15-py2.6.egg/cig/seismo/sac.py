#!/usr/bin/env python
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#                             cig.seismo
#
# Copyright (c) 2006, California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#
#    * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
#    * Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#


class SACHeader(object):
    
    def __init__(self, numberOfSamples, initialTime, timeIncrement,
                 evenlySpaced=True, fileType=1):
        
        from array import array
        from itertools import repeat
        
        self.floats = array('f', list(repeat(-12345.0, 70)))
        self.ints = array('i', list(repeat(-12345, 40)))
        self.strings = list(repeat("-12345  ", 23))
        self.strings[1] = "-12345          " # event name

        # The time increment may be slightly different than the one in
        # a '.sac' file produced by the legacy C code, due to
        # round-off error.  (Python's value will be more accurate,
        # since it's using 'double' internally, whereas the C code
        # uses 'float' for the 'dt' calculation.)
        
        self.floats[0] = timeIncrement
        self.floats[5] = initialTime
        self.ints[6] = 6 # header version number
        self.ints[9] = numberOfSamples
        self.ints[15] = fileType
        if evenlySpaced:
            self.ints[35] = 1
        else:
            self.ints[35] = 0

        return


    def tofile(self, f):
        self.floats.tofile(f)
        self.ints.tofile(f)
        for s in self.strings:
            f.write(s)
        return


def asc2sac(asciiFile, sacFile=None):
    """Convert ASCII seismogram data to a binary SAC (Seismic Analysis
    Code) data file.

    """

    # Based on the C code by Dennis O'Neill, Lupei Zhu,
    # et. al. (1988-1996).

    from struct import pack

    if sacFile is None:
        sacFile = asciiFile + ".sac"
    
    stream = open(asciiFile, "r")
    data = ''
    
    t = []
    for i in xrange(0, 2):
        line = stream.readline()
        time, amp = line.split()
        time = float(time)
        amp = float(amp)
        t.append(time)
        data += pack('f', amp)

    npts = 2
    for line in stream:
        time, amp = line.split()
        time = float(time)
        amp = float(amp)
        data += pack('f', amp)
        npts += 1

    stream.close()

    stream = open(sacFile, "wb")
    header = SACHeader(numberOfSamples=npts,
                       initialTime=t[0],
                       timeIncrement=(t[1] - t[0]),
                       )
    header.tofile(stream)
    stream.write(data)
    stream.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print >> sys.stderr, "usage: %s [filename]" % sys.argv[0]
        sys.exit()
    asc2sac(sys.argv[1])


# end of file
