#!/usr/bin/env	python
# encoding: utf-8

import math
import sys
import Feature

# Error log file.
ERROR_LOG = open("ErrorPlatesFileIO.txt", "wb")

def isPlatesFile(fileName):
    """
    Attempts to read first header in file to see if it's in PLATES4 format.
    """

    file = open(fileName, "r")
    
    firstLine = file.readline()
    if not firstLine:
        print >>sys.stderr, "Empty file '%s'." % fileName
        return False

    try:
        int(firstLine[0:2])
        int(firstLine[2:4])
        int(firstLine[5:9])
        str(firstLine[10:])
    except Exception:
        # Unable to parse first line of header so must not be PLATES4 format.
        return False

    secondLine = file.readline()
    if not secondLine:
        # To be PLATES4 format must contain two header lines.
        return False

    try:
        int(secondLine[1:4])
        float(secondLine[5:11])
        float(secondLine[12:18])
        str(secondLine[19:21])
        int(secondLine[21:25])
        int(secondLine[26:29])
        int(secondLine[30:33])
        int(secondLine[34:39])
    except Exception:
        # Unable to parse second line of header so must not be PLATES4 format.
        return False
   
    return True;
    

def readPlatesFile(platesFileName):
    """
    Read and parse a Plates input file and return a list of features of type class Feature.
    """
    global ERROR_LOG
    features = []
    platesFile = open(platesFileName, "r")
    
    try:
        lineNumber = 0
        while True:

            # Create a new Plates feature
            newFeature = Feature.Feature()

            # Read first header line of current feature
            lineNumber = lineNumber + 1
            firstLineInHeader = platesFile.readline()
            if not firstLineInHeader:
                break

            try:
                parseFirstPlatesHeaderLine(firstLineInHeader, newFeature)
            except Exception:
                print >>sys.stderr, "Error parsing line %s in file '%s' - probably not quite " \
                      "in Plates format (try making sure the columns line up with other features "\
                      "in the same file." % (lineNumber, platesFileName)
                raise

            # Read second header line of current feature.
            lineNumber = lineNumber + 1
            secondLineInHeader = platesFile.readline()
            if not secondLineInHeader:
                raise IOError("Last feature in '%s' file is missing second line of header" % \
                              platesFileName)

            try:
                parseSecondPlatesHeaderLine(secondLineInHeader, newFeature)
            except Exception:
                print >>sys.stderr, "Error parsing line %s in file '%s' - probably not quite " \
                      "in Plates format (try making sure the columns line up with other features "\
                      "in the same file." % (lineNumber, platesFileName)
                raise

            # Parse the points in this feature
            while True:
                lineNumber = lineNumber + 1
                line = platesFile.readline()

                if not line:
                     break

                # Reached end of points marker ?
                if line.split() == ["99.0000", "99.0000", "3"]:
                    break

                # Parse current point and pen
                point = line.split()
                if len(point) != 3:
                    raise Exception("At line '%s' in file '%s' there is not a point with 3 elements!"\
                                    % (lineNumber, platesFileName))

                # Convert latitude/longitude from degrees to radians.
                latitude = math.radians(float(point[0]))
                if latitude > 0.5 * math.pi:
                    latitude = 0.5 * math.pi
                elif latitude < -0.5 * math.pi:
                    latitude = -0.5 * math.pi
                    
                # Keep longitude in the range [-pi,pi]
                longitude = math.radians(float(point[1]))
                if longitude > math.pi:
                    longitude -= 2 * math.pi
                elif longitude < -math.pi:
                    longitude += 2 * math.pi
                
                pen = int(point[2])
                if pen == 2:
                    draw = True
                else:
                    draw = False

                # PLATES4 format does not make use of the point attributes so just pass in an empty list.
                newFeature.appendPoint( Feature.FeaturePoint(latitude, longitude, draw, []) )

            # Add current feature to list of features if it has enough points to form one segment
            if len(newFeature.points) >= 2:
                features.append(newFeature)
            else:
                print >>ERROR_LOG, "Feature at line %s in file '%s' has less than "\
                      "two points - ignoring feature!" % (lineNumber, platesFileName)

    finally:
        platesFile.close()

    return features


def parseFirstPlatesHeaderLine(line, feature):
    """
    Parses "line" as the first line of PLATES4 header and inserts parsed data into "feature".
    """
    
    # Read all data first in case an exception is thrown due to inability to parse "line".
    regionNumber = int(line[0:2])
    referenceNumber = int(line[2:4])
    stringNumber = int(line[5:9])
    geographicDescription = str(line[10:]).rstrip('\r\n')
    #raise IOError("test")
    
    # Now write the parsed data into "feature".
    feature.regionNumber = regionNumber
    feature.referenceNumber = referenceNumber
    feature.stringNumber = stringNumber
    feature.geographicDescription = geographicDescription


def parseSecondPlatesHeaderLine(line, feature):
    """
    Parses "line" as the second line of PLATES4 header and inserts parsed data into "feature".
    """
    
    # Read all data first in case an exception is thrown due to inability to parse "line".
    plateId = int(line[1:4])
    ageOfAppearance = float(line[5:11])
    ageOfDisappearance = float(line[12:18])
    dataTypeCode = str(line[19:21])
    dataTypeCodeNumber = int(line[21:25])
    conjugatePlateId = int(line[26:29])
    colorCode = int(line[30:33])
    numberOfPoints = int(line[34:39])
    #raise IOError("test")
    
    # Now write the parsed data into "feature".
    feature.plateId = plateId
    feature.ageOfAppearance = ageOfAppearance
    feature.ageOfDisappearance = ageOfDisappearance
    feature.dataTypeCode = dataTypeCode
    feature.dataTypeCodeNumber = dataTypeCodeNumber
    feature.conjugatePlateId = conjugatePlateId
    feature.colorCode = colorCode
    feature.numberOfPoints = numberOfPoints


def writePlatesFile(platesFileName, features):
    """
    Format and write a Plates output file from a list of features of type class Feature.
    """
    global ERROR_LOG
    
    platesFile = open(platesFileName, "wb")
    
    try:
        lineNumber = 0
        
        for feature in features:

            # Don't write feature if it has less than two points.
            if len(feature.points) < 2:
                print >>ERROR_LOG, "Feature '%s' has less than two points - not writing to "\
                      "output file '%s'!" % (feature.geographicDescription.rstrip(), platesFileName)
                continue

            # Format and write first line of header.
            firstHeaderLine = formatFirstPlatesHeaderLine(feature)
            print >>platesFile, '%s' % firstHeaderLine
            lineNumber = lineNumber + 1
  
            # Format and write second line of header.
            secondHeaderLine = formatSecondPlatesHeaderLine(feature)
            print >>platesFile, '%s' % secondHeaderLine
            lineNumber = lineNumber + 1
                
            # Output the points.
            for point in feature.points:
                
                # Set pen to 2 if drawing from previous point to current point (otherwise 3).
                if point.draw: pen = 2
                else: pen = 3
                
                # Format and write current point (convert lat/lon to degrees).
                print >>platesFile, '%9.4f %9.4f %1u' % \
                      (math.degrees(point.latitude), math.degrees(point.longitude), pen)
                lineNumber = lineNumber + 1

            # Write terminator point.
            platesFile.write("  99.0000   99.0000 3\n")
            lineNumber = lineNumber + 1
         
    finally:
        platesFile.close()


def formatFirstPlatesHeaderLine(feature):
    """
    Formats first PLATES4 header line from data in "feature" and returns as a string.
    """

    firstHeaderLine = '%2u%2u %4u %s' % \
          (feature.regionNumber,\
           feature.referenceNumber,\
           feature.stringNumber,\
           feature.geographicDescription\
           )
    
    return firstHeaderLine


def formatSecondPlatesHeaderLine(feature):
    """
    Formats second PLATES4 header line from data in "feature" and returns as a string.
    """

    secondHeaderLine = ' %3u %6.1f %6.1f %2s%4u %3u %3u %5u' % \
          (feature.plateId,\
           feature.ageOfAppearance,\
           feature.ageOfDisappearance,\
           feature.dataTypeCode,\
           feature.dataTypeCodeNumber,\
           feature.conjugatePlateId,\
           feature.colorCode,\
           len(feature.points))
    
    return secondHeaderLine
