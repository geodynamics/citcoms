#!/usr/bin/env	python
# encoding: utf-8

import math
import sys
import Feature
import PlatesFileParser

# Error log file.
ERROR_LOG = open("ErrorGMTFileIO.txt", "wb")


def isGMTFile(fileName):
    """
    Attempts to read first header in file to see if it's in GMT format.
    """

    file = open(fileName, "r")
    
    firstLine = file.readline()
    if not firstLine:
        print >>sys.stderr, "Empty file '%s'." % fileName
        return False
    
    return isGMTHeaderLine(firstLine)


def isGMTHeaderLine(line):
    """
    Returns True if "line" is a GMT header line (ie, begins with '>').
    """
    
    return line[0] == '>'
    

class GMTFileReader:
    """
    Reads lines from a file and enables peeking at a line.
    So one or more "peekline()"s followed by a "readline()" will all return same line.
    """
    
    def __init__(self, filename):
        self.lineNumber = 0
        self.__haveLine = False
        self.__line = None
        self.__file = open(filename, "r")
    
    def close(self):
        if self.__file:
            self.__file.close()
        self.__file = None
        
    def readline(self):
        self.peekline()
        self.__haveLine = False
        return self.__line
    
    def peekline(self):
        if not self.__haveLine:
            self.__line = self.__file.readline()
            self.lineNumber = self.lineNumber + 1
            self.__haveLine = True
        
        return self.__line


def readGMTFile(GMTFileName):
    """
    Read and parse a GMT input file and return a list of features of type class Feature.
    """
    
    global ERROR_LOG
    
    features = []
    GMTFile = GMTFileReader(GMTFileName)
    
    try:
        while True:
        
            # If EOF then exit feature reading loop.
            if not GMTFile.peekline():
                break
            
            # Feature to be read from file.
            newFeature = Feature.Feature()
            
            readGMTFeatureHeader(GMTFile, newFeature)

            # Parse the points in this feature
            isFirstPointInFeature = True
            numAttributes = None
            while True:
                line = GMTFile.peekline()

                # If no more lines left to read or we've come across header
                # of next feature then exit point reading loop.
                if not line or isGMTHeaderLine(line):
                    break
                
                # Commit the reading of line.
                GMTFile.readline()

                # Parse current point and pen
                point = line.split()
                if len(point) < 2:
                    raise Exception("At line '%s' in file '%s' there is not a point with at least 2 elements!"\
                                    % (lineNumber, GMTFileName))
                    
                # Keep longitude in the range [-pi,pi]
                longitude = math.radians(float(point[0]))
                if longitude > math.pi:
                    longitude -= 2 * math.pi
                elif longitude < -math.pi:
                    longitude += 2 * math.pi

                # Convert latitude/longitude from degrees to radians.
                latitude = math.radians(float(point[1]))
                if latitude > 0.5 * math.pi:
                    latitude = 0.5 * math.pi
                elif latitude < -0.5 * math.pi:
                    latitude = -0.5 * math.pi

                # Any parameters remaining after longitude and latitude are considered floating-point attributes of current point.
                attributes = [float(point[i]) for i in range(2,len(point))]
                
                # Make sure each point of current feature has the same number of attributes.
                if numAttributes:
                    if len(attributes) != numAttributes:
                        raise IOError("Number of attributes is not constant in feature "\
                                "(see line '%d' in '%s')" % (GMTFile.lineNumber, GMTFileName))
                else:
                    numAttributes = len(attributes)
   
                draw = not isFirstPointInFeature
             
                newFeature.appendPoint( Feature.FeaturePoint(latitude, longitude, draw, attributes) )
                
                isFirstPointInFeature = False

            # Add current feature to list of features if it has enough points to form one segment
            if len(newFeature.points) >= 2:
                features.append(newFeature)
            else:
                print >>ERROR_LOG, "Feature at line %s in file '%s' has less than "\
                      "two points - ignoring feature!" % (GMTFile.lineNumber, GMTFileName)

    finally:
        GMTFile.close()

    return features


def readGMTFeatureHeader(GMTFile, newFeature):
    """
    Reads GMT format header from file "GMTFile" into data members of "newFeature".
    Tries to read PLATES4 format header embedded in GMT header.
    """
    
    # Read the GMT header.
    while True:
        # See if line is part of GMT header.
        headerLine = GMTFile.peekline()
        if not headerLine or not isGMTHeaderLine(headerLine):
            break;
        
        # Commit the reading of line.
        GMTFile.readline()
        
        # Attempt to read second line of a Plates4 header after the '>' marker.
        # NOTE: attempt to parse second header line first since it is more restrictive (ie, the second header line
        # will be parsed correctly when looking for first header line but not vice versa).
        try:
            PlatesFileParser.parseSecondPlatesHeaderLine(headerLine[1:], newFeature)
        except Exception:
            # If didn't parse then simply wasn't second line of PLATES header or maybe this feature does not
            # have a PLATES header in which case no header information will be stored for current feature.
            
            # Attempt to read first line of a Plates4 header after the '>' marker.
            try:
                PlatesFileParser.parseFirstPlatesHeaderLine(headerLine[1:], newFeature)
            except Exception:
                # If didn't parse then simply wasn't first line of PLATES header or maybe this feature does not
                # have a PLATES header in which case no header information will be stored for current feature.
                pass


def writeGMTFile(GMTFileName, features):
    """
    Format and write a Plates output file from a list of features of type class Feature.
    """
    global ERROR_LOG
    
    GMTFile = open(GMTFileName, "wb")
    
    try:
        lineNumber = 0
        
        for feature in features:
            
            # Don't write feature if it has less than two points.
            if len(feature.points) < 2:
                print >>ERROR_LOG, "Feature '%s' has less than two points - not writing to "\
                      "output file '%s'!" % (feature.geographicDescription.rstrip(), GMTFileName)
                continue

            # Format and write first line of header.
            # Add the '>' marker at beginning of line since we're writing GMT format.
            firstHeaderLine = PlatesFileParser.formatFirstPlatesHeaderLine(feature)
            print >>GMTFile, '>%s' % firstHeaderLine
            lineNumber = lineNumber + 1
  
            # Format and write second line of header.
            # Add the '>' marker at beginning of line since we're writing GMT format.
            secondHeaderLine = PlatesFileParser.formatSecondPlatesHeaderLine(feature)
            print >>GMTFile, '>%s' % secondHeaderLine
            lineNumber = lineNumber + 1
                
            # Output the points.
            for point in feature.points:
                # Format and write current point (convert lat/lon to degrees).
                lineData = '%f %f' % (math.degrees(point.longitude), math.degrees(point.latitude))
                
                # Add the point attributes.
                for attribute in point.attributes:
                    lineData += ' %f' % attribute
                
                print >>GMTFile, lineData
                lineNumber = lineNumber + 1
         
    finally:
        GMTFile.close()
