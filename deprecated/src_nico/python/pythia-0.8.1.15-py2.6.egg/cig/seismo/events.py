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


class CMTSolution(object):

    def __init__(self):

        self.dataSource = ''
        self.year = 0000
        self.month = 00
        self.day = 00
        self.hour = 00
        self.minute = 00
        self._second = 00.0
        self.sourceLatitude = 0.0
        self.sourceLongitude = 0.0
        self.sourceDepth = 0.0
        self.sourceMB = 0.0
        self.sourceMs = 0.0
        self.regionName = ""
        
        self.eventName = ''
        self.timeShift = 0.0
        self.halfDuration = 0.0
        self.latitude = 0.0
        self.longitude = 0.0
        self.depth = 0.0
        self.Mrr = 0.0
        self.Mtt = 0.0
        self.Mpp = 0.0
        self.Mrt = 0.0
        self.Mrp = 0.0
        self.Mtp = 0.0


    def createFromDBModel(cls, model):
        cmtSolution = CMTSolution()
        cmtSolution.dataSource = model.dataSource.name
        cmtSolution.year = model.when.year
        cmtSolution.month = model.when.month
        cmtSolution.day = model.when.day
        cmtSolution.hour = model.when.hour
        cmtSolution.minute = model.when.minute
        cmtSolution._second = float(model.when.second) + (float(model.microsecond) / 1000000.0)
        cmtSolution.sourceLatitude = model.sourceLatitude
        cmtSolution.sourceLongitude = model.sourceLongitude
        cmtSolution.sourceDepth = model.sourceDepth
        cmtSolution.sourceMB = model.sourceMB
        cmtSolution.sourceMs = model.sourceMs
        cmtSolution.regionName = model.region.name
        cmtSolution.eventName = model.eventName
        cmtSolution.timeShift = model.timeShift
        cmtSolution.halfDuration = model.halfDuration
        cmtSolution.latitude = model.latitude
        cmtSolution.longitude = model.longitude
        cmtSolution.depth = model.depth
        cmtSolution.Mrr = model.Mrr
        cmtSolution.Mtt = model.Mtt
        cmtSolution.Mpp = model.Mpp
        cmtSolution.Mrt = model.Mrt
        cmtSolution.Mrp = model.Mrp
        cmtSolution.Mtp = model.Mtp
        return cmtSolution
    createFromDBModel = classmethod(createFromDBModel)


    def parse(cls, data):
        cmtList = []
        cmtSolution = None
        for line in data.splitlines():
            tokens = line.split(':')
            if len(tokens) == 1:
                if tokens[0]:
                    cmtSolution = CMTSolution()
                    tokens = tokens[0].split()
                    cmtSolution.dataSource = tokens[0]
                    cmtSolution.year = int(tokens[1])
                    cmtSolution.month = int(tokens[2])
                    cmtSolution.day = int(tokens[3])
                    cmtSolution.hour = int(tokens[4])
                    cmtSolution.minute = int(tokens[5])
                    cmtSolution._second = float(tokens[6])
                    cmtSolution.sourceLatitude = float(tokens[7])
                    cmtSolution.sourceLongitude = float(tokens[8])
                    cmtSolution.sourceDepth = float(tokens[9])
                    cmtSolution.sourceMB = float(tokens[10])
                    cmtSolution.sourceMs = float(tokens[11])
                    cmtSolution.regionName = ' '.join(tokens[12:])
            elif len(tokens) == 2:
                attrName = tokens[0]
                s = attrName.split()
                if len(s) > 1:
                    attrName = s[0]
                    for n in s[1:]:
                        attrName += n.capitalize()
                attrValue = tokens[1]
                if attrName == 'eventName':
                    attrValue = attrValue.strip()
                else:
                    attrValue = float(attrValue)
                setattr(cmtSolution, attrName, attrValue)
                if attrName == 'Mtp':
                    cmtList.append(cmtSolution)
                    cmtSolution = None
        return cmtList
    parse = classmethod(parse)

    def _getMrrStr2f(self): return "%.2f" % (self.Mrr * 1.0e-26)
    def _getMttStr2f(self): return "%.2f" % (self.Mtt * 1.0e-26)
    def _getMppStr2f(self): return "%.2f" % (self.Mpp * 1.0e-26)
    def _getMrtStr2f(self): return "%.2f" % (self.Mrt * 1.0e-26)
    def _getMrpStr2f(self): return "%.2f" % (self.Mrp * 1.0e-26)
    def _getMtpStr2f(self): return "%.2f" % (self.Mtp * 1.0e-26)

    def _getMinuteStr(self): return "%02d" % self.minute
    def _getSecond(self): return int(self._second)
    def _getSecondStr(self): return "%02d" % self.second
    def _getMicrosecond(self): return int((self._second - float(int(self._second))) * 1000000)

    mrr = property(_getMrrStr2f)
    mtt = property(_getMttStr2f)
    mpp = property(_getMppStr2f)
    mrt = property(_getMrtStr2f)
    mrp = property(_getMrpStr2f)
    mtp = property(_getMtpStr2f)
    
    minuteStr = property(_getMinuteStr)
    second = property(_getSecond)
    secondStr = property(_getSecondStr)
    microsecond = property(_getMicrosecond)
    
    def __str__(self):
        return """%(dataSource)s %(year)d %(month)2d %(day)2d %(hour)2d %(minute)2d %(_second)5.2f %(sourceLatitude)8.4f %(sourceLongitude)9.4f %(sourceDepth)5.1f %(sourceMB)3.1f %(sourceMs)3.1f %(regionName)s
event name: %(eventName)11s
time shift: %(timeShift)11.4f
half duration: %(halfDuration)8.4f
latitude: %(latitude)13.4f
longitude: %(longitude)12.4f
depth: %(depth)16.4f
Mrr: %(Mrr)18e
Mtt: %(Mtt)18e
Mpp: %(Mpp)18e
Mrt: %(Mrt)18e
Mrp: %(Mrp)18e
Mtp: %(Mtp)18e
""" % self.__dict__
        
    
# end of file
