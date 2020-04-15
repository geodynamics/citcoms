#!/usr/bin/env	python
# encoding: utf-8

class FeaturePoint:
    """
    Contains latitude and longitude in radians.
    Also contains Plates4 pen-down boolean (down=True and up=False).
    Also optionally contains a list of attributes each of which is a single
    floating point variable - this is primarily used by the GMT format features.
    """
    def __init__(self, latitude, longitude, draw, attributes):
        self.latitude = latitude
        self.longitude = longitude
        self.draw = draw
        self.attributes = attributes


class Feature:
    """
    Class Feature contains feature attributes read from or to be written to Plates or GMT format file.

    Class attributes:
        points - list of (latitude, longitude, draw) tuples where lat/lon in radians and
           draw is True/False if pen is down/up when moving to current point.
    """
    def __init__(self, regionNumber=0, referenceNumber=0, stringNumber=0,
                 geographicDescription="", plateId=0, ageOfAppearance=0,
                 ageOfDisappearance=0, dataTypeCode="", dataTypeCodeNumber=0,
                 conjugatePlateId=0, colorCode=0):
        self.regionNumber = regionNumber
        self.referenceNumber = referenceNumber
        self.stringNumber = stringNumber
        self.geographicDescription = geographicDescription
        self.plateId = plateId
        self.ageOfAppearance = ageOfAppearance
        self.ageOfDisappearance = ageOfDisappearance
        self.dataTypeCode = dataTypeCode
        self.dataTypeCodeNumber = dataTypeCodeNumber
        self.conjugatePlateId = conjugatePlateId
        self.colorCode = colorCode
        self.points = []

    def copyHeader(self, rhsPlatesFeature):
        self.regionNumber = rhsPlatesFeature.regionNumber
        self.referenceNumber = rhsPlatesFeature.referenceNumber
        self.stringNumber = rhsPlatesFeature.stringNumber
        self.geographicDescription = rhsPlatesFeature.geographicDescription
        self.plateId = rhsPlatesFeature.plateId
        self.ageOfAppearance = rhsPlatesFeature.ageOfAppearance
        self.ageOfDisappearance = rhsPlatesFeature.ageOfDisappearance
        self.dataTypeCode = rhsPlatesFeature.dataTypeCode
        self.dataTypeCodeNumber = rhsPlatesFeature.dataTypeCodeNumber
        self.conjugatePlateId = rhsPlatesFeature.conjugatePlateId
        self.colorCode = rhsPlatesFeature.colorCode

    # Returns a string concatentation of all fields except points  and number of points.
    def getHeaderString(self):
        return "%s %s %s %s %s %s %s %s %s %s %s" % (\
            self.regionNumber,\
            self.referenceNumber,\
            self.stringNumber,\
            self.geographicDescription,\
            self.plateId,\
            self.ageOfAppearance,\
            self.ageOfDisappearance,\
            self.dataTypeCode,\
            self.dataTypeCodeNumber,\
            self.conjugatePlateId,\
            self.colorCode\
            )
    
    def appendPoint(self, point):
        self.points.append(point)
