#!/usr/bin/env	python
# encoding: utf-8

import sys
import PlatesFileParser
import GMTFileParser

class FeatureFileType:
    """
    Type of feature file.
    """
    
    UNKNOWN, PLATES4, GMT = range(3)
    
    def __init__(self, type):
        self.type = type
    
    def __eq__(self, other):
        return self.type == other.type
    
    def __ne__(self, other):
        return self.type != other.type


def readFeaturesFile(featuresFileName):
    """
    Read and parse a Plates or GMT input file and return a list of features of type class Feature.
    This method attempts to determinet the type of feature file and delegates to appriopriate
    file parser.
    Returns a tuple containing the list of features and the file type.
    """
    
    features = []
    featureFileType = FeatureFileType(FeatureFileType.UNKNOWN)

    if PlatesFileParser.isPlatesFile(featuresFileName):
        features = PlatesFileParser.readPlatesFile(featuresFileName)
        featureFileType = FeatureFileType(FeatureFileType.PLATES4)
    elif GMTFileParser.isGMTFile(featuresFileName):
        features = GMTFileParser.readGMTFile(featuresFileName)
        featureFileType = FeatureFileType(FeatureFileType.GMT)
    else:
        raise IOError("'%s' doesn't appear to be Plates or GMT format." % featuresFileName)
    
    return (features, featureFileType);


def writeFeaturesFile(featuresFileName, features, featureFileType):
    """
    Writes features to specified file using specified file format.
    """
    
    if featureFileType == FeatureFileType(FeatureFileType.PLATES4):
        PlatesFileParser.writePlatesFile(featuresFileName, features)
    elif featureFileType == FeatureFileType(FeatureFileType.GMT):
        GMTFileParser.writeGMTFile(featuresFileName, features)
    else:
        raise IOError("Internal error - unexpected feature file type.")
