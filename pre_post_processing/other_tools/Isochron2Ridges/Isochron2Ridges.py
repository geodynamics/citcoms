#!/usr/bin/env	python
# encoding: utf-8

import sys
import math
import os.path
from optparse import OptionParser
import Feature
import FileIO
import IsochronFunctions


# The number of (age,plateId,conjugatePlateId) groups that fail the comparison of ridge distances
# for switched plateId<->conjugatePlateId.
NUM_GROUPS_FAILED_VALIDATION = 0

# Error log file.
ERROR_LOG_FILENAME = "ErrorRidgeScript.txt"
ERROR_LOG = open(ERROR_LOG_FILENAME, "wb")

# Maximum allowed deviation of ridge and conjugate ridge distances (1.0 is 100%).
MAX_DEVIATION_CONJUGATE_RIDGE_DISTANCES = 0.1


__usage__ = "%prog [options] [-h --help] input_isochron_filename output_ridges_filename"
__version__ = "0.9.4"
__description__ = "Converts isochrons to ridges and calculates ridge distances (see '%s' "\
                  "for a list of error/warning messages." % ERROR_LOG_FILENAME



class CorrectedRidges:
    def __init__(self):
        # Create a dictionary with (key,value) = (ridge header, ridge)
        self.correctedRidges = {}

    def readCorrectedRidgesFile(corrected_ridges_filename):
        # Parse Plates file to get list of corrected ridges
        correctedRidgesList = []
        correctedRidgesList = FileIO.readFeaturesFile(corrected_ridges_filename)

        for correctedRidge in correctedRidgesList:
            correctedRidges[correctedRidge.getHeaderString()] = correctedRidge

    def makeRidgeCorrection(self, ridge):
        # If there is a corrected ridge whose header matches 'ridge's header
        # then return corrected ridge.
        ridgeHeader = ridge.getHeaderString()
        if self.correctedRidges.has_key(ridgeHeader):
            return self.correctedRidges[ridgeHeader]
        else:
            return ridge



class InvertedFaultsAndRidges:
    def __init__(self):
        self.invertedList = []
    
    def readInvertedTransformFaultsAndRidgesFile(self, inverted_filename):
        """
        Reads list of plateId,conjugatePlateId pairs one per line from specified file.
        Format per line is:
           <plateId>  <conjugatePlateId>
        Returns list of pairs.
        """
        
        invertedFile = open(inverted_filename, "r")
        try:
            for line in invertedFile.readlines():
                lineSplit = line.split()

                if len(lineSplit) != 2:
                    raise ValueError

                plateIdPair = (int(lineSplit[0]), int(lineSplit[1]))
                self.invertedList.append(plateIdPair)

        except Exception:
            print >>sys.stderr, "Format of '%s' file is incorrect (should be two integer plate ids per line separated by whitespace)" %\
                  inverted_filename
            invertedFile.close()
            raise

    def shouldInvertTransformFaultsAndRidges(self, plateId, conjugatePlateId):
        return (((plateId, conjugatePlateId) in self.invertedList) or\
            ((conjugatePlateId, plateId) in self.invertedList))



class IsochronTableEntry:
    def __init__(self, isochron=None, ridge=None, ridgeDistance=0, transformFault=None, transformFaultDistance=0):
        self.isochron = isochron
        self.ridge = ridge
        self.ridgeDistance = ridgeDistance
        self.transformFault = transformFault
        self.transformFaultDistance = transformFaultDistance



def createIsochronTable(isochrons, correctedRidges, invertedFaultsAndRidges):
    """
    For each unique isochron calculate approx direction, determine ridges/transform-faults in isochron,
    correct any ridges (from corrected ridges file) and calculate ridges/transform-faults distances.
    """
    isochronTable = []
    for isochron in isochrons:

        # Differentiate between ridges and transform faults in isochron if it belongs to a
        # (plateId,conjugatePlateId) pair that requires transform-fault/ridge inversion.
        invertTransformFaultsAndRidges = invertedFaultsAndRidges.shouldInvertTransformFaultsAndRidges(\
            isochron.plateId, isochron.conjugatePlateId)

        (ridge, transformFault) = IsochronFunctions.calculateRidgeAndTransformFaultFromIsochron(\
            isochron, invertTransformFaultsAndRidges)

        # Lookup ridge from list of corrected ridges and override if a corrected version exists.
        ridge = correctedRidges.makeRidgeCorrection(ridge)

        # Calculate distance along ridge.
        ridgeDistance = IsochronFunctions.calculateFeatureDistance(ridge)

        # Calculate distance along transform fault.
        transformFaultDistance = IsochronFunctions.calculateFeatureDistance(transformFault)

        # Add info generated so far to a table.
        isochronTable.append( IsochronTableEntry(isochron, ridge, ridgeDistance, transformFault, transformFaultDistance) )
        
    return isochronTable



class IsochronListTableEntry:
    def __init__(self):
        self.isochronList = []
        self.ridgeList = []
        self.ridgeDistance = 0
        self.transformFaultList = []
        self.transformFaultDistance = 0



def createIsochronAgePlateIdConjPlateIdGroups(isochronTable):
    """
    Group isochrons with same (age, plateID, conjugatePlateID) and determine total
    ridge/transform-fault distance per group.
    Is a dictionary with key = (age, plateID, conjugatePlateID) and value = IsochronListTableEntry.
    """
    isochronGroupTable = { }
    for isochronEntry in isochronTable:

        # Assemble group key.
        groupKey = (isochronEntry.isochron.ageOfAppearance,\
                    isochronEntry.isochron.plateId,\
                    isochronEntry.isochron.conjugatePlateId)

        # Initialise dictionary entry if first using current group key.
        if not isochronGroupTable.has_key(groupKey):
            isochronGroupTable[groupKey] = IsochronListTableEntry();

        # Add isochron and corresponding ridge/transform-fault to group.
        isochronGroupTable[groupKey].isochronList.append(isochronEntry.isochron)
        isochronGroupTable[groupKey].ridgeList.append(isochronEntry.ridge)
        isochronGroupTable[groupKey].transformFaultList.append(isochronEntry.transformFault)
        
        # Add current isochron ridge and transform fault distances to group total.
        isochronGroupTable[groupKey].ridgeDistance = isochronGroupTable[groupKey].ridgeDistance +\
                                                     isochronEntry.ridgeDistance
        isochronGroupTable[groupKey].transformFaultDistance = isochronGroupTable[groupKey].transformFaultDistance +\
                                                              isochronEntry.transformFaultDistance

    return isochronGroupTable



def validateConjugateGroupRidgeDistances(age, plateId, conjugatePlateId, groupTableEntry, conjugateGroupTableEntry):
    global ERROR_LOG
    global MAX_DEVIATION_CONJUGATE_RIDGE_DISTANCES

    # Difference in group distances and maximum group distance.
    distanceDelta = math.fabs(groupTableEntry.ridgeDistance - conjugateGroupTableEntry.ridgeDistance)
    maxRidgeDistance = max(groupTableEntry.ridgeDistance, conjugateGroupTableEntry.ridgeDistance)

    # Group might not have any ridges in it.
    if maxRidgeDistance == 0:
        print >>ERROR_LOG, "Zero ridge distance for (Age=%.1f, PlateId=%s, ConjugatePlateId=%s)" %\
              (age, plateId, conjugatePlateId)
        # Failed validation.
        return False
        
    # If difference / maxDistance is too high then distances deviate too much.
    elif distanceDelta > MAX_DEVIATION_CONJUGATE_RIDGE_DISTANCES * maxRidgeDistance:
        # Generate filenames to write all ridges belonging to conjugate groups.
        groupRidgeFileName = "ValidationErrorAge%sPlate%sConjPlate%s.dat" % \
                             (age, plateId, conjugatePlateId)

        # Report error.
        print >>ERROR_LOG, "Group ridge distance mismatch %.1fpercent - "\
              "ridges file '%s' contains ridges/conjugate_ridges for age=%.1f, "\
              "plateId=%s, conjugatePlateId=%s' (in Plates format)." %\
              (100 * distanceDelta / maxRidgeDistance,\
               groupRidgeFileName,\
               age, plateId, conjugatePlateId)

##        # Shallow copy both lists into one list.
##        allRidges = []
##        allRidges.extend(groupTableEntry.ridgeList)
##        allRidges.extend(conjugateGroupTableEntry.ridgeList)
##
##        # Output both ridge files.
##        PlatesFileParser.writePlatesFile(groupRidgeFileName, allRidges)

        # Failed validation.
        return False

    # Successful validation.
    return True



def createIsochronAgeTable(isochronAgePlateIdConjPlateIdTable):
    """
    The total distance for isochrons at each age while removing duplicate
    isochron/conjugate-isochrons.
    Is a dictionary with key = (age) and value = IsochronListTableEntry.
    Functions returns (isochronAgeTable, <num_unique_conjugate_isochrons_for_all_ages>)
    """
    global NUM_GROUPS_FAILED_VALIDATION
    global ERROR_LOG
    
    isochronAgeTable = { }

    #
    # Verify that total ridge distance roughly matches for pairs of groups that have
    # switched plateID/conjugatePlateID.
    #
    # Calculate total distance for isochrons at each age (removing conjugate duplicates).
    #

    # A set of group keys (age, plateId, conjugatePlateId) to avoid conjugate duplication.
    groupsLookedAt = set()

    # Iterate over the isochron group distances (each group has same age, plateId and
    # conjugatePlateId).
    for groupKey, groupTableEntry in isochronAgePlateIdConjPlateIdTable.iteritems():
        
        # The current group might already have been compared with its conjugate.
        # If so then skip.
        if groupKey not in groupsLookedAt:
            
            # Create conjugate group key from current group key.
            (ageOfAppearance, plateId, conjugatePlateId) = groupKey
            conjugateGroupKey = (ageOfAppearance, conjugatePlateId, plateId)
            
            # Initialise dictionary entry if first using current age key.
            if not isochronAgeTable.has_key(ageOfAppearance):
                isochronAgeTable[ageOfAppearance] = IsochronListTableEntry();

            # Add group isochrons and corresponding ridges/transform-faults to group.
            isochronAgeTable[ageOfAppearance].isochronList.extend(groupTableEntry.isochronList)
            isochronAgeTable[ageOfAppearance].ridgeList.extend(groupTableEntry.ridgeList)
            isochronAgeTable[ageOfAppearance].transformFaultList.extend(groupTableEntry.transformFaultList)

            # Use maximum of group and conjugate group ridge/transform-fault distances when determining
            #total ridge/transform-fault distance for each age.
            maxGroupRidgeDistance = groupTableEntry.ridgeDistance
            maxGroupTransformFaultDistance = groupTableEntry.transformFaultDistance
           
            # If conjugate group exists.
            if isochronAgePlateIdConjPlateIdTable.has_key(conjugateGroupKey):

                # Get the group table entry for conjugate group.
                conjugateGroupTableEntry = isochronAgePlateIdConjPlateIdTable[conjugateGroupKey]
                
                # Validate the ridge distances are approximately the same.
                if not validateConjugateGroupRidgeDistances(ageOfAppearance, plateId, conjugatePlateId,\
                                                                              groupTableEntry, conjugateGroupTableEntry):
                    NUM_GROUPS_FAILED_VALIDATION = NUM_GROUPS_FAILED_VALIDATION + 1

                # Add group isochrons and corresponding ridges to group.
                isochronAgeTable[ageOfAppearance].isochronList.extend(conjugateGroupTableEntry.isochronList)
                isochronAgeTable[ageOfAppearance].ridgeList.extend(conjugateGroupTableEntry.ridgeList)
                isochronAgeTable[ageOfAppearance].transformFaultList.extend(conjugateGroupTableEntry.transformFaultList)

                # Add maximum of conjugate distances to the total distance for all isochrons
                # at age 'ageOfAppearance'. Since the conjugate pair only get visited once
                # we are not duplicating.
                if conjugateGroupTableEntry.ridgeDistance > maxGroupRidgeDistance:
                    maxGroupRidgeDistance = conjugateGroupTableEntry.ridgeDistance
                if conjugateGroupTableEntry.transformFaultDistance > maxGroupTransformFaultDistance:
                    maxGroupTransformFaultDistance = conjugateGroupTableEntry.transformFaultDistance
                
                # Add the conjugate group to the list of groups we've looked at so we don't
                # look at this conjugate pair again.
                groupsLookedAt.add(conjugateGroupKey)

            else:
                print >>ERROR_LOG, "No ridges in conjugate group (Age=%s,PlateId=%s,ConjugatePlateId=%s)" % conjugateGroupKey
            
            # Add current isochron ridge/transform-fault distance to group total.
            isochronAgeTable[ageOfAppearance].ridgeDistance = isochronAgeTable[ageOfAppearance].ridgeDistance + maxGroupRidgeDistance
            isochronAgeTable[ageOfAppearance].transformFaultDistance = isochronAgeTable[ageOfAppearance].transformFaultDistance +\
                                                                       maxGroupTransformFaultDistance

    numUniqueConjugateIsochronsForAllAges = len(groupsLookedAt)

    return (isochronAgeTable, numUniqueConjugateIsochronsForAllAges)



def denselySampleFeatureDataPoints(features, maxSpacingGreatCircleArcDistanceInMetres):
    denselySampledFeatures = []
    
    for feature in features:
        denselySampledFeatures.append(\
            IsochronFunctions.denselySampleDataPoints(feature, maxSpacingGreatCircleArcDistanceInMetres))
        
    return denselySampledFeatures



def main():
    global NUM_GROUPS_FAILED_VALIDATION
    global ERROR_LOG_FILENAME
    global ERROR_LOG
    
    # Set the ERROR_LOG in the IsochronFunctions module.
    IsochronFunctions.setErrorLogFile(ERROR_LOG)

    # Parse the command-line options.    
    parser = OptionParser(usage = __usage__,
                          version = __version__,
                          description = __description__)
    # Add option for an input plates format file containing corrected ridges.
    parser.add_option("-c", "--corrected_ridges", action="store", type="string",\
                      dest="corrected_ridges_filename",\
                      help="optional plates/GMT file containing corrected ridges to override those "\
                        "generated from input isochron file")
    parser.add_option("--output_ridges_foreach_age", action="store_true",\
                      dest="outputRidgesForEachAge",\
                      help="generate a file for all ridges at each age (same format as input file)")
    parser.add_option("--invert_faults_and_ridges", action="store", type="string",\
                      dest="invert_faults_and_ridges_filename",\
                      help="optional text file containing a list "\
                      "of plateId/conjugatePlateId pairs whose faults and ridges should be inverted")
    parser.add_option("--calculate_fault_distances", action="store_true",\
                      dest="calculateTransformFaultDistances",\
                      help="calculates transform fault distances instead of ridge distances")
    parser.add_option("--output_extra_ridge_points", action="store", type="float",\
                      dest="max_spacing_great_circle_arc_distance_in_kms",\
                      help="generates extra data points in output ridges file at a maximum great "\
                        "circle arc sample spacing (in Kms)")

    # Parse command-line options.
    (options, args) = parser.parse_args()
    if len(args) != 2:
        parser.error("incorrect number of arguments")
    inputIsochronFilename = args[0]
    outputRidgesFilename = args[1]
    
    # Parse Plates/GMT file to get a list of features
    (features, inputFeatureFileType) = FileIO.readFeaturesFile(inputIsochronFilename)
    
    # Make the output file format the same as input file format.
    outputFeatureFileType = inputFeatureFileType

    # List of ridges that will need to be corrected.
    correctedRidges = CorrectedRidges()
    if options.corrected_ridges_filename:
        correctedRidges.readCorrectedRidgesFile(options.corrected_ridges_filename)

    # List of (plateId, conjugatePlateId) pairs where faults and ridges are to be inverted.
    invertedFaultsAndRidges = InvertedFaultsAndRidges()
    if options.invert_faults_and_ridges_filename:
        invertedFaultsAndRidges.readInvertedTransformFaultsAndRidgesFile(options.invert_faults_and_ridges_filename)

    # Extract a unique list of isochrons from the general features
    uniqueIsochrons = IsochronFunctions.getUniqueIsochrons(features)

    # For each unique isochron calculate approx direction, determine ridges/transform-faults in isochron,
    # correct any ridges (from corrected ridges file) and calculate ridges/transform-faults distances.
    isochronTable = createIsochronTable(uniqueIsochrons, correctedRidges, invertedFaultsAndRidges)

    # Group isochrons with same (age, plateID, conjugatePlateID) and determine total
    # ridge/transform-fault distance per group.
    # Is a dictionary with key = (age, plateID, conjugatePlateID) and value = IsochronListTableEntry.
    isochronAgePlateIdConjPlateIdTable = createIsochronAgePlateIdConjPlateIdGroups(isochronTable)
    
    # The total distance for isochrons at each age while removing duplicate
    # isochron/conjugate-isochrons.
    # Is a dictionary with key = (age) and value = IsochronListTableEntry.
    # Functions returns (isochronAgeTable, <num_unique_conjugate_isochrons_for_all_ages>)
    (isochronAgeTable, numUniqueConjugateIsochronsForAllAges) = createIsochronAgeTable(\
        isochronAgePlateIdConjPlateIdTable)

    # Iterate over age table in order of increasing age.
    allRidges = []
    for age, isochronAgeTableEntry in sorted(isochronAgeTable.items()):
        
        # Print out the total isochron ridge distances for each age in Kms.
        if options.calculateTransformFaultDistances:
            print "Age = %s Ma, TransformFaultDistance = %.1f Km" % (age, isochronAgeTableEntry.transformFaultDistance/1000)
        else:
            print "Age = %s Ma, RidgeDistance = %.1f Km" % (age, isochronAgeTableEntry.ridgeDistance/1000)

        # Write the generated ridge features for the current age to a Plates/GMT format output file.
        if options.outputRidgesForEachAge:
            output_ridges_age_filename = os.path.join(os.path.dirname(outputRidgesFilename),\
                                                 "RidgesAtAge%.1f.dat" % age)
            FileIO.writeFeaturesFile(output_ridges_age_filename, isochronAgeTableEntry.ridgeList, outputFeatureFileType)

        # Keep a list of all ridges of all ages.
        allRidges.extend(isochronAgeTableEntry.ridgeList)

    # Sample the output ridge points more densely.
    if options.max_spacing_great_circle_arc_distance_in_kms:
        maxSpacingGreatCircleArcDistanceInMetres = options.max_spacing_great_circle_arc_distance_in_kms * 1000
        denselySampledAllRidges = denselySampleFeatureDataPoints(allRidges, maxSpacingGreatCircleArcDistanceInMetres)
    else:
        denselySampledAllRidges = allRidges
    
    # Write the generated ridge features of all ages to a Plates/GMT format output file.
    FileIO.writeFeaturesFile(outputRidgesFilename, denselySampledAllRidges, outputFeatureFileType)

    # Print some statistics.
    if numUniqueConjugateIsochronsForAllAges:
        print "%.1f percent of (age,plateId,conjugatePlateId) groups failed conjugate ridge distance comparison "\
              "for switched plateId<->conjugatePlateId (see '%s')." %\
              (100 * NUM_GROUPS_FAILED_VALIDATION / numUniqueConjugateIsochronsForAllAges, ERROR_LOG_FILENAME)

        
if __name__ == "__main__":
	main()
