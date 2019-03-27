#!/usr/bin/python
# encoding: utf-8

import math
import sys
import Feature


# Earth radius in metres.
EARTH_RADIUS_METRES = 6365000

# Maximum allowed deviation of a ridge from the strike of an isochron (in degrees).
MAX_RIDGE_STRIKE_DEVIATION_DEGREES = 45


def setErrorLogFile(errorLogFile):
    global ERROR_LOG
    ERROR_LOG = errorLogFile


    
def getUniqueIsochrons(features):
    """
    Returns list of unique isochrons (unique if have same Plates header).
    """
    uniqueIsochrons = {}
    for feature in features:
        # Is this feature an isochron ?
        if feature.dataTypeCode == "IS" or feature.dataTypeCode == "IC" or \
           feature.dataTypeCode == "IM" or feature.dataTypeCode == "RI":
            header = feature.getHeaderString()
            if not uniqueIsochrons.has_key(header):
                uniqueIsochrons[header] = feature;
            else:
                # Seems that some isochrons with same header can have different points.
                # Assume that the points can be added provided they're not exactly the same points.
                isochron = uniqueIsochrons[header]
                pointsMatch = False
                if len(isochron.points) == len(feature.points):
                    for pointIndex in range(0, len(isochron.points)):
                        if isochron.points[pointIndex] != feature.points[pointIndex]:
                            break;
                    else:
                        pointsMatch = True
                        
                if pointsMatch:
                    print >>ERROR_LOG, "Duplicate isochron detected '%s'" % header
                else:
                    isochron.points.extend(feature.points)
                
    # Return as a list.
    return uniqueIsochrons.values()



def getUnitVector3D(lat, lon):
    x = math.cos(lat) * math.cos(lon)
    y = math.cos(lat) * math.sin(lon)
    z = math.sin(lat)
    
    return (x,y,z)



def dotProduct(latStart, lonStart, latEnd, lonEnd):
    # Vector from earth centre to start surface point.
    xStart, yStart, zStart = getUnitVector3D(latStart, lonStart)
    
    # Vector from earth centre to end surface point.
    xEnd, yEnd, zEnd = getUnitVector3D(latEnd, lonEnd)
    
    dot = xStart * xEnd + yStart * yEnd + zStart * zEnd
    
    return dot



def distanceBetween(latStart, lonStart, latEnd, lonEnd):
    """
    Haversine formula is numerically more accurate than "Law of cosines" for small angles.
    http://en.wikipedia.org/wiki/Haversine_formula
    """
    global EARTH_RADIUS_METRES
    
    dlat = latEnd - latStart
    dlon = lonEnd - lonStart
    haversineLat = math.sin(0.5 * dlat)
    haversineLat = haversineLat * haversineLat
    haversineLon = math.sin(0.5 * dlon)
    haversineLon = haversineLon * haversineLon
    h = haversineLat + math.cos(latStart) * math.cos(latEnd) * haversineLon
    c = 2 * math.atan2(math.sqrt(h), math.sqrt(1-h))
    segmentDistance = EARTH_RADIUS_METRES * c
    
##            # Use inverse cosine of dot product between two vectors from earth centre to start/end of segment
##            dotp =\
##                math.cos(latStart) * math.cos(lonStart) * math.cos(latEnd) * math.cos(lonEnd) +\
##                math.cos(latStart) * math.sin(lonStart) * math.cos(latEnd) * math.sin(lonEnd) +\
##                math.sin(latStart) * math.sin(latEnd)
##            # Sometimes creeps just above 1 (enough to throw off the inverse cosine.
##            if dotp > 1:
##                dotp = 1
##            segmentDistance = math.acos(dotp) * EARTH_RADIUS_METRES
    
    return segmentDistance



def convertUnitVector3dToLatLon(x, y, z):
    if z > 1:
        z = 1
    elif z < -1:
        z = -1
    lat = math.asin(z)

    # If both x and y are zero then we cannot determine the longitude so just set it arbitrary.
    # All longitudes map to same point at the poles (which is were we're at).
    if x == 0 and y == 0:
        lon = 0
    else:
        lon = math.atan2(y, x)
    
    return (lat, lon)



def denselySampleDataPoints(feature, maxSpacingGreatCircleArcDistanceInMetres):
    denselySampledFeature = Feature.Feature()
    denselySampledFeature.copyHeader(feature)
    
    # Add start point to densely sampled feature.
    if len(feature.points) > 0:
        start = feature.points[0]
        denselySampledFeature.appendPoint( \
                Feature.FeaturePoint(start.latitude, start.longitude, start.draw, start.attributes[:]) )
        
    for pointIndex in range(1, len(feature.points)):
        
        # Get position of start and end of current feature segment.
        start = feature.points[pointIndex-1]
        end = feature.points[pointIndex]

        # If segment exists then see if need to add extra points.
        if end.draw:
            # If distance between segment end points is larger than maximum point spacing then insert extra points.
            segmentDistance = distanceBetween( \
                    start.latitude, start.longitude, end.latitude, end.longitude)
            numExtraPoints = int(segmentDistance / maxSpacingGreatCircleArcDistanceInMetres)
            if numExtraPoints > 0:
                # Start and end unit vector of current segment.
                xStart, yStart, zStart = getUnitVector3D(start.latitude, start.longitude)
                xEnd, yEnd, zEnd = getUnitVector3D(end.latitude, end.longitude)
                
                # Interpolation parameters - make the points evenly spaced inside the current segment.
                interp_increment = 1.0 / (numExtraPoints + 1)
                interp = interp_increment
                for extraPointIndex in range(numExtraPoints):
                    # Linearly interpolate between the segment end vectors.
                    x = (1 - interp) * xStart + interp * xEnd
                    y = (1 - interp) * yStart + interp * yEnd
                    z = (1 - interp) * zStart + interp * zEnd
                    
                    # Normalise linearly interpolated vector to make it unit length again.
                    vector_len = math.sqrt(x*x + y*y + z*z)
                    x /= vector_len
                    y /= vector_len
                    z /= vector_len
                    
                    # Convert 3D vector to latitude/longitude.
                    lat, lon = convertUnitVector3dToLatLon(x, y, z)
                    
                    # Linearly interpolate between the start and end point attributes.
                    assert(len(start.attributes) == len(end.attributes))
                    numAttributes = len(start.attributes)
                    interpolatedAttributes = [((1 - interp) * start.attributes[i] + interp * end.attributes[i]) \
                            for i in range(numAttributes)]
                    
                    # Add new sample point to feature.
                    denselySampledFeature.appendPoint( \
                            Feature.FeaturePoint(lat, lon, True, interpolatedAttributes) )
                    
                    interp += interp_increment

        # Add end point of current segment.
        denselySampledFeature.appendPoint( \
                Feature.FeaturePoint(end.latitude, end.longitude, end.draw, end.attributes[:]) )
    
    return denselySampledFeature



def calculateStrike(latStart, lonStart, latEnd, lonEnd):
    #
    # Simple strike calculation using (lat,lon) points fails near the poles.
    #
##    # Detect longitude wraparound from 360 degrees to 0 degrees.
##    if lonEnd - lonStart > math.pi:
##        lonEnd = lonEnd - 2 * math.pi
##    elif lonStart - lonEnd > math.pi:
##        lonStart = lonStart - 2 * math.pi
##
##    return math.atan2(latEnd - latStart, lonEnd - lonStart)

    #
    # Now use unit plane normal of plane formed by three points
    #  (1) earth centre,
    #  (2) start surface point,
    #  (3) end surface point
    # Unit normal is normalised cross product of these three points.
    #

    # Vector from earth centre to start surface point.
    xStart = math.cos(latStart) * math.cos(lonStart)
    yStart = math.cos(latStart) * math.sin(lonStart)
    zStart = math.sin(latStart)
    
    # Vector from earth centre to end surface point.
    xEnd = math.cos(latEnd) * math.cos(lonEnd)
    yEnd = math.cos(latEnd) * math.sin(lonEnd)
    zEnd = math.sin(latEnd)

    # Cross product of previous two vectors.
    xPlaneNormal = yStart * zEnd - yEnd * zStart
    yPlaneNormal = xStart * zEnd - xEnd * zStart
    zPlaneNormal = xStart * yEnd - xEnd * yStart

    # Length of cross product vector.
    planeNormalLength = math.sqrt(xPlaneNormal * xPlaneNormal +\
                                  yPlaneNormal * yPlaneNormal +\
                                  zPlaneNormal * zPlaneNormal)

    # If length is zero then both start and end surface points were equal.
    # Return zero to signal this.
    if planeNormalLength == 0:
        return (0, 0, 0)

    # Return normalised plane normal.
    return (xPlaneNormal / planeNormalLength,\
            yPlaneNormal / planeNormalLength,\
            zPlaneNormal / planeNormalLength)



def calculateIsochronStrike(isochron):
    start = isochron.points[0]
    end = isochron.points[-1]
    return calculateStrike(start.latitude, start.longitude, end.latitude, end.longitude)



def calculateRidgeAndTransformFaultFromIsochron(isochron, invertTransformFaultAndRidge):
    global MAX_RIDGE_STRIKE_DEVIATION_DEGREES
    
    isochronStrike = calculateIsochronStrike(isochron)

    # Create a new ridge feature with same header as isochron.
    ridge = Feature.Feature()
    ridge.copyHeader(isochron)
    # Change to data type to 'ridge'.
    ridge.dataTypeCode = "RI"
    
    # Create a new transform fault feature with same header as isochron.
    transformFault = Feature.Feature()
    transformFault.copyHeader(isochron)
    # Change to data type to 'transform fault'.
    transformFault.dataTypeCode = "TF"

    # Three different types of segment.
    segmentTypeNone, segmentTypeTransformFault, segmentTypeRidge = range(3)

    lastSegmentType = segmentTypeNone
    for pointIndex in range(1, len(isochron.points)):
        
        # Get position of start and end of current isochron segment.
        start = isochron.points[pointIndex-1]
        end = isochron.points[pointIndex]

        # Calculate strike of current segment and compare with strike for entire isochron.
        if end.draw:
            #
            # See if current segment belongs to ridge or transform fault.
            #

            # Calculate segment strike.
            segmentStrike = calculateStrike(start.latitude, start.longitude, end.latitude, end.longitude)

            # Calculate 3d vector dot product between two strikes (strikes are really unit plane normals).
            xSegmentStrike, ySegmentStrike, zSegmentStrike = segmentStrike
            xIsochronStrike, yIsochronStrike, zIsochronStrike = isochronStrike
            dotProduct = xSegmentStrike * xIsochronStrike +\
                         ySegmentStrike * yIsochronStrike +\
                         zSegmentStrike * zIsochronStrike;
            
            # Due to floating-point imprecision may be slightly larger than one (which throws acos()).
            if dotProduct > 1:
                dotProduct = 1
            
            #if math.degrees(math.fabs(segmentStrike - isochronStrike)) < MAX_RIDGE_STRIKE_DEVIATION_DEGREES:
            
            if dotProduct == 0:
                # We've got two segment points that are the same so don't output a segment.
                currentSegmentType = segmentTypeNone

            # The cosine of angle between to plane normals equals their vector dot product.
            elif math.degrees(math.acos(dotProduct)) < MAX_RIDGE_STRIKE_DEVIATION_DEGREES:
                currentSegmentType = segmentTypeRidge
                
            else:
                currentSegmentType = segmentTypeTransformFault
            
            # If transform faults and ridges should be inverted wrt each other...
            if invertTransformFaultAndRidge:
                if currentSegmentType == segmentTypeRidge:
                    currentSegmentType = segmentTypeTransformFault
                else:
                    currentSegmentType = segmentTypeRidge

            # If current segment belongs to a ridge...
            if currentSegmentType == segmentTypeRidge:
                if lastSegmentType != segmentTypeRidge:
                    ridge.appendPoint( Feature.FeaturePoint(start.latitude, start.longitude, False, start.attributes[:]) )
                ridge.appendPoint( Feature.FeaturePoint(end.latitude, end.longitude, True, end.attributes[:]) )

            # If current segment belongs to a transform fault...
            if currentSegmentType == segmentTypeTransformFault:
                if lastSegmentType != segmentTypeTransformFault:
                    transformFault.appendPoint( Feature.FeaturePoint(start.latitude, start.longitude, False, start.attributes[:]) )
                transformFault.appendPoint( Feature.FeaturePoint(end.latitude, end.longitude, True, end.attributes[:]) )
                
        else:
            ERROR_LOG.write("Unexpected pen up in middle of isochron '%s'!\n" % isochron.getHeaderString())
            currentSegmentType = segmentTypeNone;

        lastSegmentType = currentSegmentType

    return (ridge, transformFault)



def calculateFeatureDistance(feature):
    distance = 0
    for pointIndex in range(1, len(feature.points)):
        start = feature.points[pointIndex-1]
        end = feature.points[pointIndex]
        if end.draw:
            segmentDistance = distanceBetween(start.latitude, start.longitude, end.latitude, end.longitude)
            distance = distance + segmentDistance

    return distance

