#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Python Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Dan J. Bower, Mark Turner
#
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                        ALL RIGHTS RESERVED
#=====================================================================
'''This module holds general purpose functions for use with the Geodynamic Framework.'''

#=====================================================================
#=====================================================================
import datetime, os, pprint, re, subprocess, string, sys, traceback, math, copy
import numpy as np
import Core_GMT, random
import bisect

# load the system defaults
sys.path.append( os.path.dirname(__file__) + '/geodynamic_framework_data/')
import geodynamic_framework_configuration_types

#=====================================================================
# Global variables

# turn verbose off by default; client code can call Core_Util.verbose = True
verbose = False 

# the character uses in GMT style .xy files to indicate a header line
GMT_HEADER_CHAR = '>'

# common place to hold the earth's radius in km
earth_radius = 6371.0

# The base string of the Geodyamic Framework configuation file name 
gdf_conf = 'geodynamic_framework_defaults.conf'

# The system wide path to the base Geodyamic Framework configuation file name 
sys_gdf_conf = os.path.abspath( os.path.dirname(__file__) ) + \
                '/geodynamic_framework_data/' + \
                gdf_conf
#====================================================================
#====================================================================
#====================================================================
from html.parser import HTMLParser
from html.entities import name2codepoint

class iodp_table_parser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)
        self.inLink = False
        self.lasttag = None
        self.lastname = None
        self.lastvalue = None
        self.data_list = []
        self.href = ''

    def handle_starttag(self, tag, attrs):
        #print ("Encountered a start tag:", tag)
        self.inLink = False
        if tag == 'a':
            self.inLink = True
            for name, value in attrs:
                if name == 'href':
                    self.href = value
                    self.lasttag = tag

    def handle_endtag(self, tag):
        #print ("Encountered an end tag :", tag)
        if tag == "a":
            self.inlink = False

    def handle_data(self, data):
        #print ("Encountered some data  :", data)
        data = data.strip()
        data = data.rstrip()

        if self.lasttag == 'a' and self.inLink:
            self.data_list.append( data )
            self.data_list.append( self.href ) 

        else :
            self.data_list.append( data )

    def get_data(self):
        return self.data_list
#====================================================================
#====================================================================
#====================================================================
def cart2spher_vector(u1,u2,u3,x1,x2,x3):
    '''Convert a cartesian vector to a sphereical vector'''

    x = float(x1)
    y = float(x2)
    z = float(x3)

    r = np.sqrt(x*x + y*y + z*z)
    eps = 2.220446049250313e-16
    xy = max(eps * r, np.sqrt(x*x + y*y))

    T11 = x / r
    T21 = x * z / (xy * r)
    T31 = -y / xy
    T12 = y / r
    T22 = y * z / (xy * r)
    T32 = x / xy
    T13 = z / r
    T23 = -xy / r
    T33 = 0.

    v1 = T11*float(u1) + T12*float(u2) + T13*float(u3)
    v2 = T21*float(u1) + T22*float(u2) + T23*float(u3)
    v3 = T31*float(u1) + T32*float(u2) + T33*float(u3)

    return v1, v2, v3
#====================================================================
#====================================================================
#====================================================================
def spher2cart_vector(vr, vtheta, vphi, r, theta, phi):
    '''Convert a spherical vector to a cartesian vector'''

    sinTheta = math.sin(theta)
    cosTheta = math.cos(theta)
    
    sinPhi = math.sin(phi)
    cosPhi = math.cos(phi)

    xout=[0]*3

    xout[0] = sinTheta*cosPhi*vr + cosTheta*cosPhi*vtheta - sinPhi*vphi
    xout[1] = sinTheta*sinPhi*vr + cosTheta*sinPhi*vtheta + cosPhi*vphi
    xout[2] = cosTheta*vr - sinTheta*vtheta

    return xout[0], xout[1], xout[2]
#end

#====================================================================
#====================================================================
def cart2spher_coord(x1,x2,x3):
    '''Convert a set of cartesian coordinates to spherical coordinates'''

    s1 = np.sqrt(float(x1)*float(x1) + float(x2)*float(x2) + float(x3)*float(x3))
    s2 = np.arctan2(np.sqrt(float(x1)*float(x1) + float(x2)*float(x2)),float(x3))
    s3 = np.arctan2(float(x2),float(x1))

    return s1, s2, s3
    
#====================================================================
#====================================================================
#====================================================================
def spher2cart_coord(s1,s2,s3):
    '''Convert a set of spherical coordinates to cartesian coordinates'''

    x1 = s1 * np.sin(s2) * np.cos(s3)
    x2 = s1 * np.sin(s2) * np.sin(s3)
    x3 = s1 * np.cos(s2)

    return x1, x2, x3

#====================================================================
#====================================================================
def sph2cart(lat, lon, r=1):
    """Convert spherical coordinates to cartesian. Default raduis is 1 (unit
length). Input is in degrees, output is in km."""
    lat, lon = np.radians(lat), np.radians(lon)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return x,y,z
#====================================================================
#====================================================================
def cart2sph(x,y,z):
    """Convert cartesian geocentric coordinates to spherical coordinates.
Output is in degrees (for lat & lon) and whatever input units are for the
radius."""
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon), r
#====================================================================
#====================================================================
def local_coords(lat, lon, x,y,z):
    """Calculate local east,north,down components of x,y,z at lat,lon"""
    lat, lon = np.radians(lat), np.radians(lon)

    north = - np.sin(lat) * np.cos(lon) * x \
            - np.sin(lat) * np.sin(lon) * y \
            + np.cos(lat) * z

    east = - np.sin(lon) * x + np.cos(lon) * y

    down = - np.cos(lat) * np.cos(lon) * x \
           - np.cos(lat) * np.sin(lon) \
           - np.sin(lat) * z
    return east, north, down
#====================================================================
#====================================================================
def azimuth(east, north):
    """Returns azimuth in degrees counterclockwise from North given north and
east components"""
    azi = np.degrees(np.arctan2(north, east))
    azi = 90 - azi
    if azi <= 0:
        azi +=360
    return azi
#====================================================================
#====================================================================

#====================================================================
#====================================================================
def is_coord_within_silo(base_center, top_center, length_sq, radius_sq, test_point):
    #base_center, top_center, test_point: [x, y, z] (cartesian)

    dx = dy = dz = None       # vector d from base_center to top_center
    pdx = pdy = pdz = None    # vector pd from base_center to test point
    dot = dsq = None

    dx = top_center[0] - base_center[0] 
    dy = top_center[1] - base_center[1]
    dz = top_center[2] - base_center[2]

    pdx = test_point[0] - base_center[0] #vector from base_center to test point.
    pdy = test_point[1] - base_center[1]
    pdz = test_point[2] - base_center[2]

    # Dot the d and pd vectors to see if point lies behind the 
    # cylinder cap at base_center

    dot = pdx * dx + pdy * dy + pdz * dz

    # If dot is less than zero the point is behind the pt1 cap.
    # If greater than the cylinder axis line segment length squared
    # then the point is outside the other end cap at pt2.

    if( dot < 0.0):
        return -1
    elif( dot > length_sq ): # Above top_center
        hemisphere_dist_sq = None
        tdx = tdy = tdz = None  # vector pd from top_center to test point
        tdx = test_point[0] - top_center[0]
        tdy = test_point[1] - top_center[1]
        tdz = test_point[2] - top_center[2]
        
        hemisphere_dist_sq = tdx*tdx + tdy*tdy + tdz*tdz
        if(hemisphere_dist_sq > radius_sq): return -1
        else: return math.sqrt(hemisphere_dist_sq)
    else:
        # Point lies within the parallel caps, so find
        # distance squared from point to line, using the fact that sin^2 + cos^2 = 1
        # the dot = cos() * |d||pd|, and cross*cross = sin^2 * |d|^2 * |pd|^2
        # Carefull: '*' means mult for scalars and dotproduct for vectors
        # In short, where dist is pt distance to cyl axis: 
        # dist = sin( pd to d ) * |pd|
        # distsq = dsq = (1 - cos^2( pd to d)) * |pd|^2
        # dsq = ( 1 - (pd * d)^2 / (|pd|^2 * |d|^2) ) * |pd|^2
        # dsq = pd * pd - dot * dot / lengthsq
        #  where lengthsq is d*d or |d|^2 that is passed into this function 

        # distance squared to the cylinder axis:

        dsq = (pdx*pdx + pdy*pdy + pdz*pdz) - dot*dot/length_sq

        if( dsq > radius_sq ):
            return -1
        else:
            return math.sqrt(math.fabs(dsq))
        #end if
    #end if
#end function

#====================================================================
#====================================================================
#====================================================================
# functions for searching SORTED data
# copied from http://docs.python.org/3.3/library/bisect.html

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError

def find_lt(a, x):
    'Find rightmost value less than x'
    i = bisect.bisect_left(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_le(a, x):
    'Find rightmost value less than or equal to x'
    i = bisect.bisect_right(a, x)
    if i:
        return a[i-1]
    raise ValueError

def find_gt(a, x):
    'Find leftmost value greater than x'
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

def find_ge(a, x):
    'Find leftmost item greater than or equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError

#====================================================================
#====================================================================
#====================================================================
def convert_lon_lat_to_gmt_g( x2, y2 ):
    '''convert lon, lat to gmt 'g' bounds [0/360/-90/90]'''

    # correct for [0/360/-90/90]
    if y2 > 90:
        y2 = 180 - y2
        x2 += 180
    if y2 < -90:
        y2 = -y2 - 180
        x2 += 180
    if x2 < 0: x2 += 360
    if x2 > 360: x2 -= 360

    return x2, y2

#====================================================================
#====================================================================
#====================================================================
def convert_coordinates( in_name, incol1, incol2,
        out_name, outcol1, outcol2, R):

    '''Convert coordinates in a multiple segment xy file with header
       data (typical GMT).  incol1, incol2 and outcol1, outcol2 must
       be a combination of "lon" and "lat".
       R is "d" for lon [-180,180] and "g" for lon [0,360].'''

    if verbose: print( now(), 'convert_coordinates:' )

    gmt_char = '>'
    error = ' columns must be a combination of "lon" and "lat"'

    # read and close input file
    in_file = open( in_name, 'r' )
    lines = in_file.readlines()
    in_file.close()

    # By default, GPlates exports an extra empty header line ('>')
    # at the end of the xy line export.  This is not required and
    # it is cumbersome to code around this extra '>' in subsequent
    # routines.  Therefore, we are going to remove the last line
    # if it is a header.
    if lines[-1].startswith( gmt_char ): lines = lines[:-1]

    out_file = open( out_name, 'w' )

    entry_list = []
    for line in lines:
        if not line.strip(): continue
        if line.startswith( gmt_char ):
            out_file.write( line )
        else:
            cols = line.split()
            if incol1 == 'lon' and incol2 == 'lat':
                lon = float(cols[0])
                lat = float(cols[1])
            elif incol2 == 'lon' and incol1 == 'lat':
                lon = float(cols[1])
                lat = float(cols[0])
            else:
                print( now(), 'ERROR: input' + error )
                sys.exit(1)
            if R == 'd' and lon > 180: lon -= 360
            elif R == 'g' and lon < 0: lon += 360
            if outcol1 == 'lon' and outcol2 == 'lat':
                col1 = lon
                col2 = lat
            elif outcol2 == 'lon' and outcol1 == 'lat':
                col1 = lat
                col2 = lon
            else:
                print( now(), 'ERROR: output' + error )
                sys.exit(1)

            out_file.write( '%(col1)s %(col2)s\n' % vars() )
       
    out_file.close()

    return out_name

#====================================================================
#====================================================================
#====================================================================
def erf( x ):
    '''Error function,
from: P. Van Halen. (1989), Electronics Letters, Vol. 25, No. 9'''

    a = []
    a.append( 0.0 )
    a.append( -2.0/np.sqrt(np.pi) )
    a.append( -6.366197121956e-1 )
    a.append( -1.027728162129e-1 )
    a.append( 1.912427299414e-2 )
    a.append( 2.401479235527e-4 )
    a.append( -1.786242904258e-3 )
    a.append( 7.336113173091e-4 )
    a.append( -1.655799102866e-4 )
    a.append( 2.116490536557e-5 )
    a.append( -1.196623630319e-6 )

    lin = a[0]+a[1]*x+a[2]*x**2+a[3]*x**3+a[4]*x**4+a[5]*x**5+\
        a[6]*x**6+a[7]*x**7+a[8]*x**8
    ans = 1-np.exp(lin)

    return ans 

#====================================================================
#====================================================================
#====================================================================
def erfc( x ):

    '''Complementary error function.'''

    return 1 - erf(x)

#====================================================================
#====================================================================
#====================================================================
def find_value_on_line( in_filename, grid, out_filename ):

    '''Find value on line (track data) using a multiple segment xy
       file with header data (typical GMT).  X is lon, Y is lat 
       (both decimal degrees).  Returns a file with the same
       construct as the input and the final column returns the track
       value.'''

    cmd = in_filename + ' -m -G' + grid
    Core_GMT.callgmt( 'grdtrack', cmd, '', '>', out_filename ) 

    return out_filename

#====================================================================
#====================================================================
#====================================================================
def flatten( l, ltypes = (list, tuple) ):

    '''Flatten multiple nested lists. See 
http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html'''

    ltype = type( l )
    l = list( l )
    i = 0
    while i < len(l):
        while isinstance( l[i], ltypes ):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1

    return ltype( l )

#====================================================================
#====================================================================
#====================================================================
def flatten_nested_structure( l ):

    '''Flatten (only) the top level of a list.  For example, [[1,2],
       [3,[4,5,6]]] becomes [1,2,3,[4,5,6]].'''

    # http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python

    return [item for sublist in l for item in sublist]

#====================================================================
#====================================================================
#====================================================================
def get_cartesian_velocity_for_point( lon, lat, vtpr ):

    '''Convert spherical velocity to Cartesian velocity.

      lon must be 0 to 360
      lat must be -90 to 90
      `vtpr' is a numpy array (1D) with the following entries:
          [0] vt - non-dim colat velocity (theta)
          [1] vp - non-dim lon velocity (phi)
          [2] vr - non-dim radial velocity (radial)'''

    # Cartesian coordinates
    # citcoms colatitude (theta) and longitude (phi)
    theta = np.radians( 90-lat )
    phi = np.radians( lon )

    # projection matrix to convert velocity
    # in r,t,p to x,y,z
    sint = np.sin( theta )
    sinf = np.sin( phi )
    cost = np.cos( theta )
    cosf = np.cos( phi )
    vtpr2xyz = np.array([[ cost*cosf, -sinf, sint*cosf],
                         [ cost*sinf,  cosf, sint*sinf],
                         [   -sint  ,   0  ,   cost   ]])
    vxyz = np.dot( vtpr2xyz, vtpr )

    # return numpy array (1D)
    return vxyz

#====================================================================
#====================================================================
#====================================================================
def get_spherical_velocity_for_point( point, vxyz ):

    '''Convert Cartesian velocity to spherical velocity.

      `point' is a numpy array (1D) with the following entries:
          [0] Cartesian x coordinate of point
          [1] Cartesian y coordinate of point
          [2] Cartesian z coordinate of point
      `vxyz' is a numpy array (1D) with the following entries:
          [0] Cartesian x velocity
          [1] Cartesian y velocity
          [2] Cartesian z velocity'''

    x = point[0]
    y = point[1]
    z = point[2]

    r = np.sqrt( x**2 + y**2 + z**2 )
    r2 = np.sqrt( x**2 + y**2 )

    # these equations can be derived from the equations for
    # spherical and Cartesian geometry - DJB
    vxyz2tpr = np.array([[ z*x/(r*r2), z*y/(r*r2), -(r2**2)/(r*r2)],
                         [-y/r2      , x/r2      ,  0    ],
                         [ x/r       , y/r       ,  z/r  ]])
    vtpr = np.dot( vxyz2tpr, vxyz )

    # return numpy array (1D)
    return vtpr

#====================================================================
#====================================================================
#====================================================================
def get_current_or_last_list_entry( list, index ):

    '''Return list[index].  If index does not exist, return last
       entry in the list.'''

    if verbose: print (now(), 'get_list_entry:')

    if len(list)>0:
        while 1:
            try: 
                val = list[index]
                break
            except IndexError:
                index -= 1 
    else:
        val = list 

    return val

#====================================================================
#====================================================================
#====================================================================
def get_distance( lon1, lat1, lon2, lat2):

    '''Returns the distance between two points on the earth.  Uses
       Haversine formula:
       see http://www.movable-type.co.uk/scripts/latlong.html'''

    lon1 = np.radians( lon1 )
    lat1 = np.radians( lat1 )
    lon2 = np.radians( lon2 )
    lat2 = np.radians( lat2 )

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + \
            np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2))**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    dist = 6371.0 * c # km

    return dist 

#====================================================================
#====================================================================
#====================================================================
def get_header_GMT_xy_file_dictionary( line ):

    '''Parse a gplates exported header line into a dictionary.

    This function reads header lines of this type,
    ">sR # name: South America trench # ... # polygon: NAZ_003_000 # use_reverse: yes"
    and fills a dictionary with key/value pairs'''

    # dictionary to hold property value data
    dict = {}

    # list to hold property / value pairs from a split on '#' char
    pairs = []
    pairs = line.split('#')

    # check for missing data and just return the empty dictionary 
    if len(pairs) == 0:
        return dict

    # Specialized processing for first element from split 
    # (the first element lacks a ':' and only has a value )

    type = pairs.pop(0) # remove the first element from the list
    type = type.strip() # strip off trailing white space
    type = type.replace('>', '') # remove > char

    # add this prop/value pair to the dictionary using the key of 'type'
    dict['type'] = type

    # continue processing the rest of prop/value pairs

    for item in pairs:

        # split on the ':' character 
        list = item.split(':')

        prop = list[0]
        prop = prop.strip() # remove white space
        prop = prop.lstrip() # remove white space

        value = list[1]
        value = value.strip() # remove white space
        value = value.lstrip() # remove white space

        dict[prop] = value

    # now return the dictionary for the requested property,
    return dict

#====================================================================
#====================================================================
#====================================================================
def get_point_normal_to_subduction_zone( x, y, dx, dy, h, polarity ):

    '''Project a point (x=lon, y=lat) a distance h normal to a
       subduction zone that has a local approximate gradient dy/dx'''

    if polarity == 'R': mul = 1 
    if polarity == 'L': mul = -1

    angle = np.arctan2( -dx, dy )
    x2 = x + mul * h * np.cos( angle )/ np.cos(np.radians(y))
    y2 = y + mul * h * np.sin( angle )

    x2, y2 = convert_lon_lat_to_gmt_g( x2, y2 )

    return x2, y2

#====================================================================
#====================================================================
#====================================================================
def get_random_lines_from_file( in_filename, out_filename, density ):

    '''Get a random selection of lines from a file.'''

    # file length
    with open( in_filename ) as f:
        for i, l in enumerate(f):
            pass
    file_length = i+1

    outfile = open( out_filename, 'w' )

    sampling = int( file_length * density )

    # store random numbers in a vector
    random_numbers = []
    for i in range( sampling ):
        random_numbers.append(random.randrange(1, file_length, 1))

    # select vel vectors at random locations
    for i in range( file_length ):
        line = infile.readline()
        if random_numbers.count(i):
            outfile.write( line )

    outfile.close()

#====================================================================
#====================================================================
#====================================================================
def sample_regular_velocity_from_file( in_filename, out_filename,
    yskip, zcount, zskip ):

    '''Regularly sub-sample a velocity xyz file produced using
       make_annulus_velocity_xyz.  N.B., the line ordering does not
       follow the usual CitcomS convention (z, then y), but is
       instead (y,then z)'''

    inarray = np.loadtxt( in_filename )
    no_of_lines = len( inarray )
    ycount = int(no_of_lines / zcount)

    out_file = open( out_filename, 'w' )
    for kk in range( zcount ):
        if not kk % zskip == 0: continue
        for jj in range( ycount ):
            if not jj % yskip == 0: continue
            lineno = jj + kk*ycount
            out_line = np.array_str(inarray[lineno])
            out_line = out_line.strip('[]') + '\n'
            out_file.write( out_line )
    out_file.close()

#====================================================================
#====================================================================
#====================================================================
def generate_random_tracers( num_tracers, outer_rad=1.0, 
    inner_rad=0.55 ):

    '''Generate num_tracers within a CitcomS spherical shell.
       Method 1 based on CitcomS/lib/Tracer_setup.c 
       generate_random_tracers.'''

    if verbose: print( now(), 'generate_random_tracers:' )

    # XXX DJB
    # currently hard-coded for full sphere, although shouldn't be
    # difficult to tweak for regional models

    # hypercube rejection (1) or direct (2)
    method = 2

    # hypercube rejection method
    # also implemented in CitcomS, intuitive, but slow in comparison
    # to direct method
    if method == 1:

        # Cartesian bounds
        xmin = ymin = zmin = -1
        xmax = ymax = zmax = 1

        sphere_vol = 4/3*np.pi*(outer_rad**3-inner_rad**3)
        cube_vol = (xmax-xmin)*(ymax-ymin)*(zmax-zmin)
        frac_vol = sphere_vol / cube_vol

        # number of tracers necessary in cube for approximately
        # num_tracers in sphere
        num_cube = np.ceil( num_tracers / frac_vol )

        # extra 0.5% to account for statistical fluctuations
        num_cube *= 1.005

        # numpy vector operations seem to be the fastest way of doing this
        # (i.e., significantly faster than looping over random numbers)
        # sample points in Cartesian cube
        x = xmin + np.random.random_sample((num_cube,)) * (xmax-xmin)
        y = ymin + np.random.random_sample((num_cube,)) * (ymax-ymin)
        z = zmin + np.random.random_sample((num_cube,)) * (zmax-zmin)

        # convert to spherical coordinates
        r = np.sqrt(x*x+y*y+z*z)
        theta = np.arccos(z/r)
        phi = np.arctan2(y,x) + np.pi

        # filter between outer_rad and inner_rad to exclude tracers that
        # fall outside the sphere
        indices = np.where(((r <= outer_rad) & (r >= inner_rad)))

        # now filter, and keep only first num_tracers entries
        r = r[indices][:num_tracers]
        theta = theta[indices][:num_tracers]
        phi = phi[indices][:num_tracers]

    # direct method
    elif method == 2:

        u1 = np.random.random_sample((num_tracers,))
        u2 = np.random.random_sample((num_tracers,))
        u3 = np.random.random_sample((num_tracers,))

        # derive by considering a spherical volume element
        # dV=r**2 sin(t) dt dp dr
        b = np.power(inner_rad,3)
        a = np.power(outer_rad,3) - b
        r = np.power( a*u1+b, 1.0/3 )
        theta = np.arccos( 1-2*u2 )
        phi = 2*np.pi*u3

    # testing
    # radius
    #r1 = 0.9 # outer shell
    #r2 = 0.6 # inner shell
    #shell_vol_r1 = 4/3*np.pi*(1**3-r1**3) # shell volume
    #shell_vol_r2 = 4/3*np.pi*(r2**3-0.55**3) # shell volume
    #out_shell = np.size(np.where(r>r1)) # num tracs in outer
    #inner_shell = np.size(np.where(r<r2)) # num tracs in inner
    #print( now(), 'outer shell tracer density=', out_shell/shell_vol_r1 )
    #print( now(), 'inner shell tracer density=', inner_shell/shell_vol_r2 )

    # theta
    #low_hemi_out = np.size(np.where(theta>np.pi/2))
    #upp_hemi_out = np.size(np.where(theta<np.pi/2))
    #hemi_exp = num_tracers/2
    #Vol = 4/3*np.pi*1**3/2
    #Vol = 4/3*np.pi*(1**3-0.55**3)/2
    #print( now(), 'lower hemisph tracer density=', low_hemi_out/Vol )
    #print( now(), 'upper hemisph tracer density=', upp_hemi_out/Vol )
    #print( now(), 'expect hemisph tracer density=', round(hemi_exp/Vol) )

    # solid angle (smaller constricts more around poles)
    #angle = np.pi/24
    #Vol = 2/3*np.pi*(1-np.cos(angle))*(1**3-0.55**3)
    # North polar tracer density
    #polar_num = np.size(np.where(theta<angle))
    #print( now(), 'North polar density=', polar_num/Vol )
    # South polar tracer density
    #polar_num = np.size(np.where(theta>np.pi-angle))
    #print( now(), 'South polar density=', polar_num/Vol )

    # CitcomS radius, colat, lon (all non-dimensional)
    return r, theta, phi

#====================================================================
#====================================================================
#====================================================================
def get_slab_center_dist( depth_km, start_depth, slab_dip, roc,
    vertical_slab_depth ):

    '''While shallow, follow circle (radius of curvature), if deeper,
       then follow a line with constant dip that tangents the
       circle.'''

    if verbose: print( now(), 'get_slab_center_dist:' )

    # branch to determine either curving slab or slab with constant dip
    if depth_km < vertical_slab_depth: depth = depth_km - start_depth
    # branch to determine vertical slab below vertical_slab_depth
    else: depth = vertical_slab_depth - start_depth

    d_c = roc * (1.0 - np.cos( slab_dip ))
    if depth <= 0: dist = 0
    if depth > 0.0 and depth <= d_c: 
        dist = np.sqrt( roc**2 - (roc-depth)**2 )
    # this branch is (also) called for vertical slabs
    # effectively, vertical_slab_depth should be > d_c
    # which I think it always will be - DJB
    elif depth > d_c: 
        x_c = roc * np.sin( slab_dip )
        x_prime = (depth - d_c) / np.tan( slab_dip )
        dist = x_c + x_prime

    # shift dist a fraction thermal thickness toward subducting plate

    # this shift looks wrong for zero depth where the slab begins
    # 'behind' the subduction zone (clearly seen in comparison to
    # the line data).

    if depth_km > 0:
        dist = (dist - 60) / 110.0 # units are equatorial degrees
    else:
        dist /= 110.0

    return dist

#====================================================================
#====================================================================
#====================================================================
def get_slab_data( arg, input_subduction_file, out_filename ):

    '''Create a file containing the pertinent header details for slab
    assimilation and the coordinates of the line features.  Assume
    default values and override by GPML header data if specified in
    the input configuration file (GPML_HEADER).  Also determine the
    maximum slab depth to use in the loop that constructs the
    temperature grids.'''

    if verbose: print( now(), 'get_slab_data:' )

    # parameters from dictionary
    age = arg['age']
    gmt_char = arg['gmt_char']
    GPML_HEADER = arg['GPML_HEADER']
    UM_depth = arg['UM_depth']

    # default values for all slabs 
    default_slab_depth = arg['default_slab_depth']
    default_slab_dip = arg['default_slab_dip']
    default_slab_LM_descent_rate = arg['slab_LM_descent_rate']
    default_slab_start_depth = 0.0 # sdepth
    default_slab_UM_descent_rate = arg['slab_UM_descent_rate']

    # N.B. For complete consistency, eventually we may want to
    # relate the descent rate of slabs in the mantle (particularly
    # upper mantle) to the velocity of the subducting plate (taken
    # from gplates data).  However, there isn't an easy way to do 
    # this at the moment so the sinking rate and plate velocity remain
    # decoupled for this part of the script - DJB

    # read and close input file
    infile = open( input_subduction_file, 'r' )
    lines = infile.readlines()
    infile.close()

    out_file = open( out_filename, 'w' )

    # store slab depths to determine the maximum
    slab_depth_l = []

    for line in lines:

        # shortcut to write out xy (coordinate) data
        if not line.startswith( gmt_char ):
            out_file.write( line )
            continue

        # exit if missing subduction zone polarities
        if line[1:3] not in ['sL','sR']:
            errorline = line.rstrip('\n')
            print( now(), errorline )
            print( now(), 'ERROR: get_slab_data: unknown subduction zone polarity')
            sys.exit(1)

        # restore default parameters
        slab_depth = default_slab_depth
        slab_dip = default_slab_dip
        slab_LM_descent_rate = default_slab_LM_descent_rate
        slab_start_depth = default_slab_start_depth
        slab_UM_descent_rate = default_slab_UM_descent_rate

        # check for values in GPML header
        if GPML_HEADER:
            if verbose: print( now(), 'reading GPML header data')
            hd = get_header_GMT_xy_file_dictionary( line )

            # override slab dip
            if hd.get('subductionZoneDeepDip', 'Unknown') != 'Unknown':
                slab_dip = float( hd['subductionZoneDeepDip'] )

            # override slab start depth
            if hd.get('slabFlatLyingDepth', 'Unknown') != 'Unknown':
                slab_start_depth = float( hd['slabFlatLyingDepth'] )

            # override slab depth
            if hd.get('subductionZoneDepth', 'Unknown') != 'Unknown':
                slab_depth = float( hd['subductionZoneDepth'] )

            # check for subduction initiation
            sz_age = hd.get('subductionZoneAge', 'Unknown')
            # DJB - previous formulation
            #TF_slab_flat = hd.get('slabFlatLying', 'Unknown')
            #if TF_slab_flat == 'Unknown' and sz_age != 'Unknown':

            # DJB - new formulation that ensures flat slab leading edges also
            # have a slab_depth that can be modified by convergence
            if sz_age != 'Unknown':
                sz_age = float( sz_age )
                if age <= sz_age:
                    # override slab UM descent rate
                    if hd.get('subductionZoneConvergence', 'Unknown') != 'Unknown':
                        slab_UM_descent_rate = float( hd['subductionZoneConvergence'])

                    # factor of 10 gives depth in km for slab_descent_rate in cm/yr
                    UM_duration = UM_depth / (slab_UM_descent_rate*10.0) # Myr
                    s_duration = sz_age - age
                    if s_duration < UM_duration:
                        slab_depth = slab_UM_descent_rate*s_duration*10.0
                    elif s_duration >= UM_duration:
                        slab_depth = UM_depth + slab_LM_descent_rate*(s_duration-UM_duration)*10.0
                else:
                    # for regional models only
                    slab_depth = 0

        # maximum (theoretical) depth of slab is the core-mantle boundary
        # XXX DJB - this hard-coding can be removed by also passing pid_d
        # into this function
        if slab_depth > 2866:
            print( now(), 'WARNING: truncating slab_depth to 2866 km (core-mantle boundary)')
            slab_depth = 2866

        slab_depth_l.append( slab_depth )

        # xy header line
        out_line = '%s DEPTH=%f DIP=%f START_DEPTH=%f\n' % (line[0:3],slab_depth,slab_dip,slab_start_depth)
        out_file.write( out_line )

    out_file.close()

    # set maximum slab depth to loop over
    arg['slab_depth_gen'] = max( slab_depth_l )

    return out_filename

#====================================================================
#====================================================================
#====================================================================
def get_stencil_depth_and_smooth( control_d, depth_km ):

    '''Determine stencil depth and stencil smoothing.  These 
       variables increase linearly during subduction initiation.'''

    if verbose: print( now(), 'get_stencil_depth_and_smooth:' )

    stencil_depth_max = control_d['stencil_depth_max']
    stencil_smooth_max = control_d['stencil_smooth_max']
    stencil_smooth_min = control_d['stencil_smooth_min']

    # grow stencil depth and smoothing with subduction initiation
    sten_depth = min( depth_km, stencil_depth_max )
    sten_smooth = depth_km / stencil_depth_max * stencil_smooth_max
    sten_smooth = min( sten_smooth, stencil_smooth_max )
    sten_smooth = max( sten_smooth, stencil_smooth_min )

    if verbose: print( now(), sten_depth, sten_smooth )

    return sten_depth, sten_smooth

#====================================================================
#====================================================================
#====================================================================
def increase_resolution_of_xy_line( in_filename, res, out_filename, Q ):

    '''Increase the resolution of line data stored in a multiple
       segment xy file with header data (typical GMT).
       X is lon, Y is lat (both decimal degrees), res in km.
       -Q argument applied to GMT project if 'Q' is True'''

    if verbose: print( now(), 'increase_resolution_of_xy_line:' )

    gmt_char = '>'
    rm_list = []

    # read and close input file
    in_file = open( in_filename, 'r' )
    lines = in_file.readlines()
    in_file.close()

    # open file (remove previous version, since appending)
    subprocess.call( 'rm ' + out_filename, shell=True)
    out_file = open( out_filename, 'ab')

    project_file = 'increase_resolution_of_xy_line.xyp'
    rm_list.append( project_file )

    for line in lines:
        # header line
        if line.startswith( gmt_char ):
            data_list = []
            out_file.write( bytes( line, 'UTF-8') )
        # coordinate line
        else:
            lon, lat = line.split()
            flon = float( lon )
            flat = float( lat )
            data_list.append( (flon, flat) )
            if len(data_list) < 2: continue
            clon, clat = data_list[-1] # current
            plon, plat = data_list[-2] # previous

            # identical points crash GMT project
            if clon == plon and clat == plat:
                continue

            cmd = '-C%(plon)s/%(plat)s -E%(clon)s/%(clat)s' % vars()
            cmd += ' -G%(res)s' % vars() # res in km
            if Q: cmd += ' -Q'
            Core_GMT.callgmt( 'project', cmd, '', '>', project_file )
            out_data = np.loadtxt( project_file )
            np.savetxt( out_file, out_data, fmt='%f %f %f' )
            
    out_file.close()

    remove_files( rm_list )

    return out_filename

#====================================================================
#====================================================================
#====================================================================
def Ierf( x ):

    '''Integral of error function,
from: P. Van Halen. (1989), Electronics Letters, Vol. 25, No. 9.'''

    c0 = 0.564189534361
    a = []
    a.append( 0.0 ) # a0
    a.append(-1.0/c0) # a1
    a.append(-5.707880967287e-1) # a2
    a.append(-8.371977808743e-2) # a3
    a.append(7.682906530457e-3) # a4
    a.append(1.639148621936e-3) # a5
    a.append(-7.630344277202e-4) # a6
    a.append(1.208254138998e-4) # a7
    a.append(-7.041324094169e-6) # a8

    lin = a[0]+a[1]*x+a[2]*x**2+a[3]*x**3+a[4]*x**4+a[5]*x**5+\
        a[6]*x**6+a[7]*x**7+a[8]*x**8
    ans = x - c0*(1-m.exp(lin))

    return ans 

#=====================================================================
#=====================================================================
#=====================================================================
def is_sequence(arg):

    return (not hasattr(arg, "strip") and
            hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__"))

#=====================================================================
#=====================================================================
#=====================================================================
def make_annulus_xyz( master, project, out_name, grid_list ):

    '''Make a GMT xyz file (dist, radius, value) containing the data
       for gridding along an arbitrary (user-specified) great circle
       path or part of a great circle path.

       master:  `master' dictionary containing various parameters
       project:  filename for project data, columns (lon, lat, dist)
       out_name: output filename
       grid_list: grids to track through'''

    if verbose: print( now(), 'make_annulus_xyz:' )

    coor_d = master['coor_d']
    radius_list = coor_d['radius']

    # store data (remove old file first if it exists)
    if os.path.isfile( out_name ):
        subprocess.call( 'rm ' + out_name, shell=True)
    out_file = open( out_name, 'ab' )
    out_track = 'make_annulus_xyz_track.xyz'

    for nn, radius_nd in enumerate( radius_list ):

        find_value_on_line( project, grid_list[nn], out_track )
        dist, value = np.loadtxt( out_track, usecols=(2,3), unpack=True, comments='>' )
        # compute polar angular offset
        pao = (dist[-1]-dist[0]) / 2 + dist[0]
        max_dist = dist[-1]

        radius_tile = np.tile( radius_nd, len(dist) )
        out_data = np.column_stack( (dist, radius_tile, value) )
        np.savetxt( out_file, out_data, fmt='%f %f %f' )

    out_file.close()

    # clean up
    subprocess.call( 'rm ' + out_track, shell=True)

    # return polar angular offset and maximum distance
    return pao, max_dist

#=====================================================================
#=====================================================================
#=====================================================================
def make_annulus_cross_section_velocity( master_d, project,
    velo_scale, vtpr_grd_files_l, out_file_names_l ):

    '''Apply polar angular offset to correct velocities to a local
       radial unit vector.  This is necessary for -JPa or -Jpa GMT
       projections.'''

    make_cross_section_velocity( master_d, project, velo_scale, 
    vtpr_grd_files_l, out_file_names_l, True )

#=====================================================================
#=====================================================================
#=====================================================================
def make_rectangle_cross_section_velocity( master_d, project, 
    velo_scale, vtpr_grd_files_l, out_file_names_l ):

    '''Do not correct azimuth using polar angular offset.  This is
       necessary for -JX or -Jx GMT projections.'''

    make_cross_section_velocity( master_d, project, velo_scale, 
    vtpr_grd_files_l, out_file_names_l, False )

#=====================================================================
#=====================================================================
#=====================================================================
def make_map_velocity( master_d, mesh, velo_scale, vtp_grid_files_l, 
    out_file_name ):

    '''Make a GMT four column file (lon, lat, azimuth, length)
       containing the data for plotting velocity vectors in map
       view.'''

    if verbose: print( now(), 'make_map_velocity:' )

    coor_d = master_d['coor_d']
    pid_d = master_d['pid_d']
    radius = pid_d['radius']
    thermdiff = pid_d['thermdiff']
    scalev = pid_d['scalev']

    out_track = 'make_map_velocity_track.xyz'

    velo_data = []
    for vcomp in vtp_grid_files_l:
        find_value_on_line( mesh, vcomp, out_track )
        velo_data.append( np.loadtxt( out_track ) )

    lon = velo_data[0][:,0]
    lat = velo_data[0][:,1]
    vx = velo_data[0][:,2]
    vy = velo_data[1][:,2]

    azimuth = np.degrees(np.arctan2(vy,-vx))
    magnitude = np.hypot(vx,vy) # units are cm/yr
    length = magnitude / velo_scale

    out_data = np.column_stack( (lon,lat,azimuth,length) )
    np.savetxt( out_file_name, out_data, fmt='%f %f %f %f' )

#=====================================================================
#=====================================================================
#=====================================================================
def make_cross_section_velocity( master_d, project, velo_scale,
    vtpr_grid_files_l, out_file_names_l, apply_pao ):

    '''Make a GMT four column file (dist, radius, azimuth, length)
       containing the data for plotting velocity vectors along an
       arbitrary (user-specified) great circle path or part of a great
       circle path.  Also make xyz files of velocity tangential to
       surface and in radial direction.  Data is only output for
       entries in out_file_names_l that do not contain an empty 
       string.  All output velocities have units of cm/yr.

       master_d:  `master_d' dictionary containing various parameters
       project:  filename for project data; columns (lon, lat, dist)
       velo_scale: velocity scale cm/yr per plotting inch
       vtpr_grid_files_l: list (ordered by nodez) of 3-tuples 
           (vt, vp, vr) for each depth, where vt is the grid name of
           citcoms theta velocity, vp is citcoms phi velocity, and vr
           is citcoms radial velocity.  All are non-dimensional
           velocity.
       out_file_names_l: output file names
           out_file_names_l[0] : velocity vectors
           out_file_names_l[1] : tangential (to surface) velocity
           out_file_names_l[2] : radial velocity
           out_file_names_l[3] : magnitude of velocity'''

    # N.B., for correct velocity polarity, the project file may need
    # to be ordered from small colat/lon to larger colat/lon
    # Still needs some testing to be sure - DJB

    if verbose: print( now(), 'make_cross_section_velocity:' )

    coor_d = master_d['coor_d']
    pid_d = master_d['pid_d']
    radius = pid_d['radius']
    radius_list = coor_d['radius']
    thermdiff = pid_d['thermdiff']
    scalev = pid_d['scalev']

    # store data
    if out_file_names_l[0]:
        velocity_file = open( out_file_names_l[0], 'w' )
    if out_file_names_l[1]:
        tangent_file = open( out_file_names_l[1], 'w' )
    if out_file_names_l[2]:
        radial_file = open( out_file_names_l[2], 'w' )
    if out_file_names_l[3]:
        vmag_file = open( out_file_names_l[3], 'w' )

    out_track = 'make_cross_section_velocity_track.xyz'

    for nn, radius_nd in enumerate( radius_list ):

        # get (non-dimensional) v_theta, v_phi, v_radius from grids
        velo_data = []
        for vcomp in vtpr_grid_files_l[nn]:
            find_value_on_line( project, vcomp, out_track )
            velo_data.append( np.loadtxt( out_track ) )

        # compute polar angular offset
        # by default, this will rotate the velocity vectors to
        # conform to -JPXXX/pao
        pao = velo_data[0][-1][2] / 2
        max_dist = velo_data[0][-1][2]

        # loop over all data and process
        velo_comp = []
        for pp in range( len(velo_data[0]) ):
            line = tuple( [velo_data[n][pp] for n in range(3)] )
            velo_comp.append( line )
            if len( velo_comp ) < 3: continue

            # current
            clon = velo_comp[-1][0][0]
            clat = velo_comp[-1][0][1]

            # previous
            plon = velo_comp[-2][0][0]
            plat = velo_comp[-2][0][1]
            if verbose: print( now(), '( lon, lat, radius )' )
            if verbose: print( now(), '(', plon, plat, radius_nd, ')' )
            dist = velo_comp[-2][0][2]
            vt = velo_comp[-2][0][3]
            vp = velo_comp[-2][1][3]
            vr = velo_comp[-2][2][3]
            vtpr = np.array( [vt, vp, vr] )
            vxyz = get_cartesian_velocity_for_point( plon, plat, vtpr )
            if verbose: print( now(), 'vxyz=', vxyz )

            # previous previous
            pplon = velo_comp[-3][0][0]
            pplat = velo_comp[-3][0][1]

            # tangential in-plane unit vector about previous
            p1 = convert_point_to_cartesian( clon, clat, radius_nd )
            p2 = convert_point_to_cartesian( pplon, pplat, radius_nd )
            v1 = (p1-p2) / np.linalg.norm( p1-p2 )
            if verbose: print( now(), 'in-plane normal=', v1 )

            # radial unit vector
            # Center of Earth is third point, i.e. (0,0,0)
            p3 = convert_point_to_cartesian( plon, plat, radius_nd )
            v2 = p3 / np.linalg.norm( p3 )
            if verbose: print( now(), 'radial normal=', v2 )

            # out-of-plane unit vector
            v3 = np.cross( v1, v2 )
            if verbose: print( now(), 'out-of-plane normal=', v3 )

            # velocity components
            ivelo = np.dot( v1, vxyz )
            rvelo = np.dot( v2, vxyz )

            if verbose: print( now(), 'ivelo=', ivelo )
            if verbose: print( now(), 'rvelo=', rvelo )

            azimuth = np.degrees(np.arctan2(ivelo,rvelo))
            if azimuth < 0: azimuth = 360 + azimuth
            if apply_pao: azimuth += dist - pao
            magnitude = np.hypot(ivelo,rvelo)*scalev
            length = magnitude / velo_scale
            if verbose: print( now(), 'magnitude=', magnitude, 'cm/yr' )
            if out_file_names_l[0]:
                out_line = '%f %f %f %f\n' % (dist, radius_nd, azimuth, length)
                velocity_file.write( out_line )

            # xyz files for gridding (output is dimensional, cm/yr)
            if out_file_names_l[1]:
                out_line = '%f %f %f\n' % (dist, radius_nd, ivelo*scalev )
                tangent_file.write( out_line )
            if out_file_names_l[2]:
                out_line = '%f %f %f\n' % (dist, radius_nd, rvelo*scalev )
                radial_file.write( out_line )
            if out_file_names_l[3]:
                out_line = '%f %f %f\n' % (dist, radius_nd, magnitude )
                vmag_file.write( out_line )

    if out_file_names_l[0]: velocity_file.close()
    if out_file_names_l[1]: tangent_file.close()
    if out_file_names_l[2]: radial_file.close()
    if out_file_names_l[3]: vmag_file.close()

    # clean up
    subprocess.call( 'rm ' + out_track, shell=True )

    # return polar angular offset and maximum distance
    return pao, max_dist

#=====================================================================
#=====================================================================
#=====================================================================
def make_link( file1, file2 ):
    '''Call the link function to create a link named file2 to an existing file1.'''
    if verbose: print( now(), 'make_link: file1 =', file1)
    if verbose: print( now(), 'make_link: file2 =', file2)
    if not os.path.exists(file2):
        cmd = 'ln -v -s %(file1)s %(file2)s' % vars()
        if verbose: print( now(), cmd )
        subprocess.call( cmd, shell=True )
    else:
        print( "WARN: ", file2, "already exists; NOT re-linking to requested file:", file1)
#=====================================================================
def make_dir( directory_name ):
    '''Make a directory.'''
    if not os.path.isdir( directory_name ):
        cmd = 'mkdir -v -p '+ directory_name
        if verbose: print( now(), cmd )
        subprocess.call( cmd, shell=True )
    else:
        print( "WARN: directory already exists; not recreating ", directory_name )
#=====================================================================
#=====================================================================
def make_flat_slab_age_depth_xyz( master, flat_slab_depth_grd,
        out_filename ):

    '''Description.'''

    if verbose: print( now(), 'make_flat_slab_age_depth_file:')

    # parameters
    control_d = master['control_d']
    func_d = master['func_d']
    afile_1 = control_d['afile_1']
    grid_dir = control_d['grid_dir']
    grd_res = str(control_d['grd_res'])
    rm_list = func_d['rm_list']
    R = control_d['R']

    # leading edge file has correct bounds, but need to correct
    # polygon file
    out_name = control_d['flat_slab_polygon_file'].split('/')[-1].rstrip('xy')
    out_name += 'g.xy'
    rm_list.append( out_name )
    control_d['flat_slab_polygon_file'] = convert_coordinates(
        control_d['flat_slab_polygon_file'], 'lon', 'lat', out_name,
        'lon', 'lat', 'g' )
    flat_slab_polygon_file = control_d['flat_slab_polygon_file']
    flat_slab_leading_file = control_d['flat_slab_leading_file']

    # mask of flat slab
    grid = grid_dir + '/flat_slab_mask.grd'
    rm_list.append( grid )
    cmd = flat_slab_polygon_file + ' -F -m -I' + grd_res + ' -R' + R
    cmd += ' -NNaN/0/1'
    Core_GMT.callgmt( 'grdmask', cmd, '', '', '-G' + grid )

    out_list = []
    for ifile in [ afile_1, flat_slab_depth_grd ]:
        out_name = ifile.rstrip('grd') + 'mask.grd'
        rm_list.append( out_name )
        cmd = ifile + ' ' + grid + ' MUL'
        Core_GMT.callgmt( 'grdmath', cmd, '', '=', out_name )
        in_name = out_name
        out_name = out_name.rstrip('grd') + 'xyz'
        rm_list.append( out_name )
        cmd = in_name + ' -S' 
        Core_GMT.callgmt( 'grd2xyz', cmd, '', '>', out_name )
        lon, lat, val = np.loadtxt( out_name, unpack=True )
        out_list.append( val )

    age = out_list[0]
    depth = out_list[1]
    np.savetxt( out_filename, np.column_stack( (lon,lat,age,depth) ),
        fmt = '%g %g %g %g' )

    return out_filename

#=====================================================================
#=====================================================================
#=====================================================================
def make_flat_slab_depth_grd( master, grd_filename ):

    '''Description.'''

    if verbose: print( now(), 'make_flat_slab_depth_grd:' )

    control_d = master['control_d']
    func_d = master['func_d']

    # parameters from dictionary
    default_depth_km = 100.0 # km XXX DJB - historical value!
    default_dip_degrees = control_d['default_slab_dip']
    flat_slab_leading_file = control_d['flat_slab_leading_file']
    gmt_char = control_d['gmt_char']
    rm_list = func_d['rm_list']
    R = control_d['R']
    sub_file = control_d['sub_file']
    tension = str(control_d['tension'])
    #grd_min = '0' # depths must be greater than zero
    grd_res = str(control_d['grd_res'])
    xyz_filename = grd_filename.rstrip('grd') + 'xyz'
    rm_list.append( xyz_filename )
    xyz_file = open( xyz_filename, 'w' )

    # (1) depths to the base of the subduction zone,
    #     i.e., where the normal slab becomes flat.
    in_file = open( sub_file )
    lines = in_file.readlines()
    in_file.close()
    for line in lines:
        # header line
        if line.startswith( gmt_char ):
            polarity = line[2]
            hd = get_header_GMT_xy_file_dictionary( line )
            szd = hd.get('subductionZoneDepth', 'Unknown')
            szdd = hd.get('subductionZoneDeepDip', 'Unknown')

            # subduction zone depth
            if szd != 'Unknown': fszd = float(szd)
            else: fszd = default_depth_km
            # maximum depth of flat slab at subduction zone
            # XXX DJB unsure if the next line is necessary
            # subduction zones may have slabs with great depths 
            # that will strongly torque the data fitting.  Since we
            # are constructing flat slabs, assign a maximum depth
            # here, regardless of what the line data implies
            #if fszd > default_depth_km: fszd = default_depth_km

            # subduction zone dip
            if szdd != 'Unknown': fszdd = float(szdd)
            else: fszdd = default_dip_degrees

            # dist in degrees
            dist = fszd / (np.tan(np.radians(fszdd))*110.0)

            data_list = []

        # coordinate data
        else:
            lon, lat, dummy = line.split()
            data_list.append( (float(lon), float(lat)) )
            if len( data_list ) < 3: continue

            clon, clat = data_list[-1] # current data
            plon, plat = data_list[-2] # previous data
            pplon, pplat = data_list[-3] # previous previous data
            dx = clon - pplon # for approx gradient about previous data
            dy = clat - pplat # for approx gradient about previous data
            nlon, nlat = get_point_normal_to_subduction_zone(
                        plon, plat, dx, dy, dist, polarity )
            out_line = '%(nlon)s %(nlat)s %(fszd)s\n' % vars()
            xyz_file.write( out_line )

    # (2) depths to the leading edge
    in_file = open( flat_slab_leading_file )
    lines = in_file.readlines()
    in_file.close()
    for line in lines:
        if line.startswith( gmt_char ):
            hd = get_header_GMT_xy_file_dictionary( line )
            fld = hd.get('slabFlatLyingDepth', 'Unknown') 
        elif fld != 'Unknown':
            lon, lat, dummy = line.split()
            xyz_file.write( '%(lon)s %(lat)s %(fld)s\n' % vars() )

    # XXX DJB - unsure if we need this or not
    # (3) depths (always zero) at the trench
    #in_file = open( sub_file )
    #lines = in_file.readlines()
    #in_file.close()
    #for line in lines:
    #    if not line.startswith( gmt_char ):
    #        lon, lat = line.split()
    #        out_line = '%(lon)s %(lat)s 0\n' % vars()
    #        xyz_file.write( out_line )

    xyz_file.close()

    # make grid
    median_xyz = xyz_filename.rstrip('xyz') + 'median.xyz'
    rm_list.append( median_xyz )
    cmd = xyz_filename + ' -I' + grd_res + ' -R' + R
    Core_GMT.callgmt( 'blockmedian', cmd, '', '>', median_xyz )
    # XXX DJB - used to also constrain grd_min and grd_max,
    # but I don't think this is necessary
    cmd = median_xyz + ' -I' + grd_res + ' -R' + R + ' -T' + tension
            #+ ' -Ll' + grd_min  + ' -Lu' + grd_max
    Core_GMT.callgmt( 'surface', cmd, '', '', ' -G' + grd_filename )
    # force pixel registration (this cannot be done using surface)
    args = '%(grd_filename)s -T' % vars()
    Core_GMT.callgmt( 'grdsample', args, '', '', '-G%(grd_filename)s' % vars() )
    
    return grd_filename

#=====================================================================
#=====================================================================
#=====================================================================
def make_flat_slab_temperature_xyz( master, kk ):

    '''Description.'''

    if verbose: print( now(), 'make_flat_slab_temperature_xyz:' )

    coor_d = master['coor_d']
    control_d = master['control_d']
    func_d = master['func_d']
    pid_d = master['pid_d']
    grid_dir = control_d['grid_dir']
    rm_list = func_d['rm_list']

    # parameters from dictionaries
    depth_km = coor_d['depth_km'][kk]
    # contains ( lon, lat, age, depth )
    flat_age_xyz = func_d['flat_age_xyz']
    fssd = control_d['flat_slab_stencil_depth']
    lith_age_min = control_d['lith_age_min']
    oceanic_lith_age_max = control_d['oceanic_lith_age_max']
    radius = pid_d['radius']
    scalet = pid_d['scalet']
    sten_max = control_d['stencil_max']
    suffix = control_d['suffix']
    temperature_mantle = control_d['temperature_mantle']
    temperature_min = control_d['temperature_min']

    in_file = open( flat_age_xyz )
    lines = in_file.readlines()
    in_file.close()

    # temperature
    temp_xyz = grid_dir + '/flattemp' + suffix + 'xyz'
    rm_list.append( temp_xyz )
    out_file = open( temp_xyz, 'w' )

    # stencil
    sten_xyz = grid_dir + '/flatsten' + suffix + 'xyz'
    rm_list.append( sten_xyz )
    out_file2 = open( sten_xyz, 'w' )

    # similar to make_slab_temperature_xyz()
    dT = (temperature_mantle - temperature_min) / 2

    for line in lines:
        lon, lat, age, depth = [float(entry) for entry in line.split()]
        age = max( age, lith_age_min )
        age = min( age, oceanic_lith_age_max )
        dd = 1 / (2 * np.sqrt( age / scalet ))
        dist = abs( depth_km - depth )* 1E3 / radius # non-dim
        temp = temperature_mantle - dT * erfc( dd*dist )
        out_file.write( '%g %g %g\n' % ( lon, lat, temp ) )
        if depth_km < fssd:
            out_file2.write( '%g %g %g\n' % ( lon, lat, sten_max ) )

    out_file.close()
    out_file2.close()

    return ( temp_xyz, sten_xyz )

#====================================================================
#====================================================================
#====================================================================
def make_great_circle_with_azimuth( lon0, lat0, azimuth,
                                         incr, L, out_name ):

    '''Make a great circle using a start point (lon0, lat0) and an
       azimuth.  incr specifies point spacing in degrees.'''

    cmd  = '-C%(lon0)s/%(lat0)s' % vars()
    cmd += ' -A%(azimuth)s' % vars()
    cmd += ' -G%(incr)s' % vars()
    cmd += ' -L%(L)s' % vars()

    Core_GMT.callgmt( 'project', cmd, '', '>', out_name )

#====================================================================
#====================================================================
#====================================================================
def make_great_circle_with_two_points( lon0, lat0, lon1, lat1,
                                         incr, L, out_name ):

    '''Make a great circle using a start point (lon0, lat0) and an
       end point (lon1, lat1).  incr is increment in degrees, and if
       L=w the great circle is constrained between the two points.'''

    cmd  = '-C%(lon0)s/%(lat0)s' % vars()
    cmd += ' -E%(lon1)s/%(lat1)s' % vars()
    cmd += ' -G%(incr)s' % vars()

    if L is not None: cmd += ' -L%(L)s' % vars()

    Core_GMT.callgmt( 'project', cmd, '', '>', out_name )

#=====================================================================
#=====================================================================
#=====================================================================
def make_pdf_from_ps_list( ps_list, filename ):

    '''Make a pdf file from a list of postscript files.  PDF toolkit
       (pdftk) must be installed for this function to work.'''

    if verbose: print( now(), 'make_pdf_from_ps_list:')

    cmd_pdftk = 'pdftk'
    rm_list = []
    for kk, ps in enumerate( ps_list ):
        if os.path.isfile( ps ):
            pdf = ps.rstrip('ps') + 'pdf'
            rm_list.append( pdf )
            cmd = ['ps2pdf', ps, pdf]
            subprocess.call( cmd ) #, shell=True )
            cmd_pdftk += ' ' + pdf

    # test if any postscripts have been located and appended
    if len( cmd_pdftk ) != 5:
        cmd_pdftk += ' cat output ' + filename
        subprocess.call( cmd_pdftk, shell=True )
        remove_files( rm_list )
    # no postscripts exist
    else:
        print( now(), 'no files exist to convert and merge to a PDF' )
        print( now(), 'aborting creation of', filename )

    return filename

#====================================================================
#====================================================================
#====================================================================
def make_slab_temperature_xyz( master, kk ):

    '''Description.'''

    if verbose: print( now(), 'make_slab_temperature_xyz:' )

    coor_d = master['coor_d']
    control_d = master['control_d']
    func_d = master['func_d']
    age = str(control_d['age'])
    grid_dir = control_d['grid_dir']
    pid_d = master['pid_d']
    rm_list = func_d['rm_list']

    # parameters from dictionaries
    depth_km = coor_d['depth_km'][kk] # km
    depth = coor_d['depth'][kk] # non-dim
    UM_depth = control_d['UM_depth']
    if depth_km < UM_depth: advection = control_d['UM_advection']
    else: advection = control_d['LM_advection']
    gmt_char = control_d['gmt_char']
    lith_age_min = control_d['lith_age_min']
    oceanic_lith_age_max = control_d['oceanic_lith_age_max']
    myr2sec = pid_d['myr2sec']
    N_slab_pts = control_d['N_slab_pts']
    radius_km = pid_d['radius_km']
    roc = control_d['radius_of_curvature']
    scalet = pid_d['scalet']
    slab_age_xyz = func_d['slab_age_xyz']
    spacing_slab_pts = control_d['spacing_slab_pts']
    stencil_depth_min = control_d['stencil_depth_min']
    stencil_width = control_d['stencil_width']
    stencil_width_smooth = control_d['stencil_width_smooth']
    temperature_mantle = control_d['temperature_mantle']
    temperature_min = control_d['temperature_min']
    thermdiff = pid_d['thermdiff']
    vertical_slab_depth = control_d['vertical_slab_depth']

    # slab temperature
    temp_xyz  = grid_dir + '/slabtemp' + control_d['suffix'] + 'xyz'
    rm_list.append( temp_xyz )
    out_file = open( temp_xyz, 'w' )

    # slab stencil
    sten_xyz = grid_dir + '/slabsten' + control_d['suffix'] + 'xyz'
    rm_list.append( sten_xyz )
    out_file2 = open( sten_xyz, 'w' )

    # degrees are `equatorial' (relevant for longitude)
    startval = -0.5*((N_slab_pts-1)*spacing_slab_pts)
    d_p_degrees = [startval+ii*spacing_slab_pts for ii in range(N_slab_pts)]
    d_p = [abs(ii*(110.0/radius_km)) for ii in d_p_degrees] # non-dim 

    if verbose: print( now(), 'd_p_degrees', d_p_degrees )
    if verbose: print( now(), 'd_p', d_p )

    for line in open( slab_age_xyz ):
        # header line
        if line.startswith( gmt_char ):
            # subduction zone polarity
            # already checked in get_slab_data(), but let us check again!
            if line[2] == 'R' or line[2] == 'L':
                polarity = line[2]
            else:
                errorline = line.rstrip('\n')
                print( now(), errorline )
                print( now(), 'ERROR: cannot determine subduction zone polarity' )
                sys.exit(1)

            line_segments = line.split(' ')
            slab_depth = float(line_segments[1].lstrip('DEPTH=') )
            slab_dip = float(line_segments[2].lstrip('DIP=') )
            slab_dip = np.radians(slab_dip) # to radians
            start_depth = float(line_segments[3].lstrip('START_DEPTH=') )
            dist = get_slab_center_dist( depth_km, start_depth, slab_dip, roc,
                       vertical_slab_depth )
            data_list = []

            # 1/2 because a boundary layer is created on each side of the slab
            # 1/sin(dip) to conserve down-dip buoyancy
            # temperature_mantle-temperature_min is temperature drop
            dT = (temperature_mantle-temperature_min) / (2*np.sin( slab_dip ))

            sten_depth, sten_smooth = \
                get_stencil_depth_and_smooth( control_d, slab_depth )
            # minimum depth for stencil for cleaner subduction
            # initiation
            sten_depth = max( sten_depth, stencil_depth_min )
            # extra 25 km to ensure all thermal anomaly is included
            max_stencil_depth = sten_depth + sten_smooth + 25

        # coordinate line
        else:
            lon, lat, dummy, age = line.split()
            data_list.append( (float(lon), float(lat), float(age)) )
            if len( data_list ) < 3 : continue

            clon, clat, cage = data_list[-1] # current data
            plon, plat, page = data_list[-2] # previous data
            pplon, pplat, ppage = data_list[-3] # previous previous data
            dx = clon - pplon # for approx gradient about previous data
            dy = clat - pplat # for approx gradient about previous data
            # grdtrack can produce some negative values due to interpolation
            # use min and max oceanic ages to ensure a positive age
            # and also be compatible with the user's parameter choice
            page = max( page, lith_age_min )
            page = min( page, oceanic_lith_age_max )
            dd = 1 / (2 *np.sqrt( page / scalet ))

            for ii in range( N_slab_pts ):

                dist2 = dist + advection*d_p_degrees[ii]
                nlon, nlat = get_point_normal_to_subduction_zone(
                                 plon, plat, dx, dy, dist2, polarity )

                # stencil
                if depth_km >= start_depth and depth_km <= max_stencil_depth:

                    # horizontal (lateral) direction
                    smooth = stencil_width_smooth/6371
                    twidth = stencil_width/2/6371
                    arg = ( d_p[ii]-twidth ) / smooth
                    sten_val = 0.5 * (1-np.tanh(arg))

                    # vertical (depth) direction
                    smooth = sten_smooth / radius_km
                    tdepth = sten_depth / radius_km
                    arg = ( depth-tdepth ) / smooth
                    sten_val *= 0.5 * (1-np.tanh(arg))

                    out_line = '%(nlon)g %(nlat)g %(sten_val)g\n' % vars()
                    out_file2.write( out_line )


                # temperature
                if depth_km >= start_depth and depth_km <= slab_depth:
                    slab_temp = temperature_mantle - dT * erfc( dd*d_p[ii] )

                    # prevent overprinting of pre-existing slabs
                    frac = (slab_temp - temperature_mantle)
                    frac /= temperature_mantle
                    frac *= 100 # to percentage
                    if frac <= -5: # more than 5% temperature contrast
                        out_line = '%(nlon)g %(nlat)g %(slab_temp)g\n' % vars()
                        out_file.write( out_line )


    out_file.close()
    out_file2.close()

    return ( temp_xyz, sten_xyz )

#=====================================================================
#=====================================================================
#=====================================================================
def make_uniform_background( filename, spacing, value ):

    '''Description.'''

    if verbose: print( now(), 'make_uniform_background:' )

    out_file = open( filename, 'w' )
    lon = 0
    while lon <= 360: 
        lat = -89
        while lat <= 89:
            out_file.write( '%g %g %g\n' % ( lon, lat, value ))
            lat += spacing * random.random()
        lon += spacing * random.random()
    out_file.close()

    return filename

#=====================================================================
#=====================================================================
#=====================================================================
def now():

    '''Redefine now() with short format yyyy-mm-dd hh:mm:ss'''

    return str(datetime.datetime.now())[11:19]

#=====================================================================
#=====================================================================
#=====================================================================
def convert_point_to_cartesian( lon, lat, rad ):

    '''Convert lon, lat, non-dim rad to Cartesian x, y, z. (CitcomS)'''

    # citcoms colatitude (theta) and longitude (phi)
    theta = np.radians( 90-lat )
    phi = np.radians( lon )

    # Cartesian coordinates
    rst = rad * np.sin( theta )
    x = rst * np.cos( phi )
    y = rst * np.sin( phi )
    z = rad * np.cos( theta )

    return np.array([x, y, z])

#=====================================================================
#=====================================================================
#=====================================================================
def convert_point_to_spherical( x, y, z ):

    '''Convert Cartesian x, y, z to r, theta, phi. (CitcomS)'''

    r = np.sqrt(x*x+y*y+z*z)
    # 0 to pi
    theta = np.arccos(z/r)
    # 0 to 2*pi
    phi = np.arctan2(y,x) + np.pi

    return np.array([r, theta, phi])

#=====================================================================
#=====================================================================
#=====================================================================
def remove_files( file_list ):

    '''Remove files.'''

    print( now(), 'remove_files:' )

    arg_l = ['rm','-rf'] + flatten( file_list )
    print( now(), ' '.join(arg_l) )
    subprocess.call( arg_l )

#=====================================================================
#=====================================================================
#=====================================================================
def get_subduction_zone_data(file):
    '''process a subduction boundary type file (e.g. 'topology_subduction_boundaries_0.00Ma.xy') into a listl.
Each subduction zone is a dictionary structure'''

    if verbose: print( now(), 'get_subduction_zone_data: START')
    if verbose: print( now(), 'get_subduction_zone_data: file =', file)

    # create an empty list of subduction zones for this file 
    sz_list = []

    # read and close input file
    infile = open( file, 'r' )
    lines = infile.readlines()
    infile.close()

    # set up a working vars
    coords = []
    sz = {}

    # process the input file 
    for line in lines:

        # check for header line
        if line.startswith( GMT_HEADER_CHAR ) :

            # check for a current working sz
            if not sz == {} :
                # complete the current working sz 
                if verbose: print( now(), 'get_subduction_zone_data: complete working sz name =', sz['header_data']['name'] )

                # copy the current working coords into the sz
                sz['coords'] = coords

                # add a deep copy of this sz to the master list
                sz_list.append( dict( sz ) )

                # reset place holder vars for next sz  
                coords = []
                sz = {}

            # check for new sz 
            if sz == {}:
                # start up a new sz

                # NOTE: Maybe store the file this sz came from, and the actual header line ?
                # sz['file'] = file
                # sz['header_line'] = line

                # get for values in GPML header
                hd = get_header_GMT_xy_file_dictionary( line )
                sz['header_data'] = hd

                # NOTE: Maybe save selected entries at top level ?
                #sz['name']  = hd.get('name', 'Unknown')
                #sz['order'] = hd.get('subductionZoneSystemOrder', 'Unknown')

                # Set a warning if missing subduction zone polarities
                if line[1:3] not in ['sL','sR']:
                    sz['warning'] = 'unknown subduction zone polarity'

                continue # to next line 

        # else this is a coordinate data line, add it to the current working coords list 
        else: 
            # process coordinate data for this sz 
            coords.append( line.strip() ) 
            continue # to next line 

    # end of loop over lines
    if verbose : print( now(), 'get_subduction_zone_data: len(sz_list) =', len(sz_list))
    if verbose : print( now(), 'get_subduction_zone_data: sz_list =')
    if verbose : tree_print( sz_list )
    return sz_list 
#=====================================================================
#=====================================================================
#=====================================================================
def get_subduction_systems( file ):
    '''From subduction zone boundary file, generate a report highlighting various data'''

    # get a list of all the subduction zones in the file 
    sz_list = get_subduction_zone_data( file )
    if verbose : print(now(), 'get_subduction_systems: total # of sz =', len(sz_list) )

    # a subset of sz's with system names
    sz_list_in_systems = []

    # set up working dictionaryies to process systems
    sz_systems = {}
    sz_systems_ordered = {}

    # the list of system names
    system_names = []

    # loop over all the sz in this file 
    for sz in sz_list:
        # get the name of the system this sz belongs to 
        system_name = sz['header_data']['subductionZoneSystem']
        if system_name == 'Unknown' :
            continue # to next sz

        # else, add this system to the main list of names
        if not system_name in system_names : system_names.append( system_name )

        # populate the subset 
        sz_list_in_systems.append( sz )

        # set up an empty list for this name
        sz_systems[system_name] = []
        sz_systems_ordered[system_name] = []

        #if verbose : print( now(), '==================================================')
        #if verbose : print( now(), 'get_subduction_systems: sz =')
        #if verbose : tree_print( sz )

    # end of loop over sz's in this list 


    if verbose : print(now(), 'get_subduction_systems: system_names =', system_names)
    if verbose : print(now(), 'get_subduction_systems: # of sz in systems =', len(sz_list_in_systems) )

    # loop over system subset to populate the un-ordered dictionary 
    for sz in sz_list_in_systems:
        # get the name of the system this sz belongs to 
        system_name = sz['header_data']['subductionZoneSystem']
        if system_name in system_names:
            sz_systems[system_name].append(sz)
    # end of loop over sz's in sublist

    #if verbose : print(now(), 'get_subduction_systems: sz_systems =')
    #if verbose : tree_print( sz_systems )

    # order each named system
    for system_name in sz_systems.keys() :

        # get an empty dict to order by subductionZoneSystemOrder value
        order_dict = {}

        # loop over all the sz's for this name 
        for sz in sz_systems[system_name] :
            # get the order 
            order = sz['header_data']['subductionZoneSystemOrder']
            if verbose : print(now(), 'get_subduction_systems: order =', order)
            order_dict[order] = sz

        # re-order this system by subductionZoneSystemOrder value
        keys = sorted(order_dict)
        for k in keys :
            sz_systems_ordered[system_name].append( order_dict[k] )
    # end of ordering loop
            
    if verbose : print(now(), 'get_subduction_systems: sz_systems_ordered =')
    if verbose : tree_print( sz_systems_ordered )

    return sz_systems_ordered
#=====================================================================
#=====================================================================
#=====================================================================
def tree_print(arg):
    '''print the arg in tree form'''
    print(now())
    pprint.PrettyPrinter(indent=2).pprint(arg)
#=====================================================================
#=====================================================================
#=====================================================================
def parse_general_one_value_per_line_type_file(filename):
    '''Parse a general ascii text file, one value per line, and return the data as a list of values.
       Any data following the # character will be ignored.
       Blank lines will be ignored.
'''
    global verbose 

    # the list to return
    ret_list = []

    # loop over lines in the file
    infile = open(filename,'r')
    for line in infile:
        if verbose: print(now(), "parse_general_one_value_per_line_type_file: line ='", line, "'")

        # skip past comments and blank lines
        if line.startswith('#'):
            continue # to next line
        if line.rstrip() == '':
            continue # to next line

        if '#' in line:
            # get and split the data
            val, com = line.split('#')
            val = val.strip() # clean up white space
            val = val.rstrip() # clean up white space
            com = com.rstrip() # clean up white space
            if verbose: print(now(), "parse_general_one_value_per_line_type_file: val=", val, '; com=', com)
        else:
            val = line.strip()
            val = val.rstrip()

        # add this entry to the dict
        ret_list.append(val)
    # end of loop over lines
    return ret_list
#=====================================================================
#=====================================================================
#=====================================================================
def parse_general_key_equal_value_linetype_file(filename):
    '''Parse a general 'key=value', one item per line, type file and return the data as a dictionary.
       NOTE: No subsections are allowed.  Use parse_configuration_file() if subsections are requried.
'''
    global verbose 

    # the dictionary to return
    ret_dict = {}

    # loop over lines in the file
    infile = open(filename,'r')
    for line in infile:
        if verbose: print(now(), "parse_general_key_equal_value_linetype_file: line ='", line, "'")

        # skip past comments and blank lines
        if line.startswith('#'):
            continue # to next line
        if line.rstrip() == '':
            continue # to next line

        # get and split the data
        key, val = line.split('=')
        key = key.strip() # clean up white space
        key = key.rstrip() # clean up white space
        val = val.strip() # clean up white space
        val = val.rstrip() # clean up white space
        
        # TODO: Dan fixed bug (commented out end='').  Mark to check
        if verbose: print(now(), "parse_general_key_equal_value_linetype_file: key=", key, '; val=', val ) #, end='')

        # add this entry to the dict
        ret_dict[key]=val
    # end of loop over lines

    return ret_dict
#=====================================================================
#=====================================================================
def parse_geodynamic_framework_defaults():
    '''Read the geodynamic_framework defaults file from of these locations:
            local: "./geodynamic_framework_defaults.conf"
            user home dir: "~/geodynamic_framework_defaults.conf"
            system defult: "/path/to/system"
       and return the data as a dictionary.

       If the local and user home files are missing, 
       then copy the system file to the current working directory.
'''

    # Alert the user as to which defaults file is being used:
    print( now(), 'Core_Util.parse_geodynamic_framework_defaults():' )

    # check for local framework defaults 
    file = './geodynamic_framework_defaults.conf'
    if os.path.exists(file):
        print(now(), 'Using the local current working directory Geodynamic Framework defaults file:\n', file)
        return parse_general_key_equal_value_linetype_file( file )

    # check for user's default file 
    file = os.path.expanduser('~/geodynamic_framework_defaults.conf')
    if os.path.exists(file):
        print(now(), 'Using the home directory Geodynamic Framework defaults file:\n', file)

        # copy the home file to the cwd so the user has a record
        # of which defaults were used
        print(now(), 'Copying the user file to the current working directory as: "geodynamic_framework_defaults.conf"')
        cmd = 'cp ' + file + ' ' + 'geodynamic_framework_defaults.conf'
        print( now(), cmd )
        subprocess.call( cmd, shell=True )

        return parse_general_key_equal_value_linetype_file( file )

    # local and user files are missing, parse the default system file:

    # check for system default file
    file = os.path.abspath( os.path.dirname(__file__) ) + "/geodynamic_framework_data/geodynamic_framework_defaults.conf"
    if os.path.exists(file):
        print(now(), 'Using the System Geodynamic Framework defaults file:\n', file)

        # copy the system file to the cwd
        print(now(), 'Copying the System file to the current working directory as: "geodynamic_framework_defaults.conf"')
        cmd = 'cp ' + file + ' ' + 'geodynamic_framework_defaults.conf'
        print( now(), cmd )
        subprocess.call( cmd, shell=True )

        return parse_general_key_equal_value_linetype_file( file )

    # halt the script if we cannot find a valid file 
    print(now(), 'Cannot find a valid geodynamic_framework defaults file.')
    print(now(), 'Please update from SVN to get /geodynamic_framework_data/geodynamic_framework_defaults.conf')
    sys.exit(1)
   
#=====================================================================
#=====================================================================
def parse_configuration_file_to_create_variable_type_relations(filename):
    '''This function reads a geodynamic framework configuration file and returns a dictionary of that data where:

the keys are variable names from the .cfg file;
the values are tuples: (data type, sample value, sample commment); 

For more info on the format of a framework configuation file, 
please see the documentation for 'parse_configuration_file()' 
and 'make_example_config_file()'

'''

    # the master dictionary to hold variable type relations
    # the keys are variable names read in from the .cfg file 
    # the values are tuples of the form (type, example, comment)
    type_dict = {}

    # get the input file data 
    in_file = open(filename,'r')

    # loop over lines and set up default entries in the type dict
    for line in in_file:

        # skip past comments and blank lines
        if line.startswith('#'):
            continue # to next line
        if line.rstrip() == '':
           continue # to next line

        # skip section lines
        if '[' in line:
            continue # to next line

        # split data line into variable, value and comment:
        var, val = line.split('=')

        # split value from comment 
        com = ''
        if ';' in val:
            val, com = val.split(';')

        # clean up white space
        var = var.strip() 
        var = var.rstrip() 
       
        val = val.strip() 
        val = val.rstrip() 

        com = com.strip() 
        com = com.rstrip() 

        # create a new tuple with an unknown type
        t = 'unk', val, com
        # add a new entry to the master dict
        type_dict[var] = t

    # end of loop over lines

    # loop over entries in master dict and determine type 
    keys = sorted(type_dict)
    for k in keys:

        # get the value and the comment 
        val = type_dict[k][1]
        com = type_dict[k][2]

        #if verbose: print("TEST: ", val, com)

        # check for types and reset the enrty in the master dict with a new tuple

        # check for empty string
        if not val:
            type_dict[k] = 'string', val, com
            continue

        # check for values with explict string wrapping 
        if val.startswith('"'):
            type_dict[k] = 'string', val, com
            continue # to next variable
        if val.startswith("'"):
            type_dict[k] = 'string', val, com
            continue # to next variable

        # check for explicit boolean values
        if val == 'True':
            type_dict[k] = 'bool', val, com
            continue # to next variable

        if val == 'true':
            type_dict[k] = 'bool', val, com
            continue # to next variable

        if val == 'False':
            type_dict[k] = 'bool', val, com
            continue # to next variable

        if val == 'false':
            type_dict[k] = 'bool', val, com
            continue # to next variable

        # check for time values ( 3:00:00 )
        if ':' in val:
            type_dict[k] = 'string', val, com
            continue # to next variable

        # check for lists
        if ',' in val:

            # separate values in list and determine list type
            values = val.split(',')

            # check for list of ints
            pattern = re.compile('^[-+]?[0-9]*$')
            match = pattern.match( values[0] )
            if match:
              type_dict[k] = 'int_list', val, com
              continue # to next variable

            # check for floats in various forms:
            # simple cases like '1.2' and '11.22'
            # and scientific notations like '1.2e+3', '-1.2e+3', etc. '1.2e3'
            pattern = re.compile('^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
            match = pattern.match( values[0] )
            if match:
                type_dict[k] = 'float_list', val, com
                continue # to next variable

            # check for strings
            pattern = re.compile('\D+')
            match = pattern.match( values[0] )
            if match:
                type_dict[k] = 'string_list', val, com
                continue # to next variable
        
        # end of checks for lists

        # check for ints: 1, -1, 10, -1234
        pattern = re.compile('^[-+]?[0-9]*$')
        match = pattern.match( val )
        if match:
            type_dict[k] = 'int', val, com
            continue # to next variable

        # check for floats in various forms:
        # simple cases like '1.2' and '11.22'
        # and scientific notations like '1.2e+3', '-1.2e+3', etc. '1.2e3'
        pattern = re.compile('^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$')
        match = pattern.match( val )
        if match:
            type_dict[k] = 'float', val, com
            continue # to next variable

        # check for paths:
        if ('/' in val):
            type_dict[k] = 'path', val, com
            continue # to next variable

        # check for strings
        pattern = re.compile('\D+')
        match = pattern.match( val )
        if match:
            type_dict[k] = 'string', val, com
            continue # to next variable

    # end of loop over master dictionay keys

    # update values from the system type map dict 
    for k in geodynamic_framework_configuration_types.type_map.keys():
        # get the type 
        type = geodynamic_framework_configuration_types.type_map[k]
        # update the auto parsed dict with a tuple: 
        # updated type, empty sample value, empty comment
        type_dict[k] = type, '', ''

    if verbose: 
        print(now(), 'parse_configuration_file_to_create_variable_type_relations: type_dict =')
        tree_print( type_dict )

    return type_dict

#=====================================================================
#=====================================================================
def parse_configuration_file( filename, update_top_level_dict_from_section=False, merge_top_level_params_into_section=False):

    '''Parse a geodynamic framework configuration file and return a dictionary of that data.
    
    A configuration file consists of lines with parameter and value pairs.

    See the 'make_example_config_file' function in this module for a simple but complete .cfg file.

    Many of the project scripts will create their own example .cfg files using the -e option.

    A valid configuration file may have any number of top level parameters, 
    of the form: 'param = value' with one pair per line.

    The top level params are followed by zero or more sub section blocks 
    containing their own parameter lines.

    Blank lines and lines starting with the '#' character are ignored as comments.

    Parameter names my be any value python string suitable for a dictionary key.
    Value types may be string, float, int, bool, or lists of those base types.

    Before populating the return dictionary, the .cfg file is parsed to determine
    the base data types.  For more info please see the function: 
      parse_configuration_file_to_create_variable_type_relations()
    These two functions work hand in hand to create a stable typed dictionary reprentation 
    of the parameters and values in the .cfg file.

    The return dictionary will contain all the top level parameter = value pairs 
    as top level key/value dictionary entries.

    Each subsection will be a nested dictionary, with all of its own parameter = value pairs
    as key/value dictionary entries.

    The return dictionary also has a special top level entry with key name of '_SECTIONS_'.
    This value is a list of the subsection names as strings. 
    This list is useful to iterate over the subsections with code such as:

       # loop over subsections in cfg dictionary 
       for s in cfg_d['_SECTIONS_'] : 
           # process subsection s dictionary as needed

    As a convenience, all the subsection parameters are also populated into the top level dictionary.  
    This allows for a single dictionary containing all the data in a .cfg file. 

    In general it is best to have unique parameters across top level and all subsections.

    Two options control the case where a common paramter name appears in both the 
    top level and in one or more subsections.

    If this function is called with 'update_top_level_dict_from_section = True',
    then the common parameter value will be set by the last subsection using 
    that common parameter name.

    If this function is called with 'merge_top_level_params_into_section = True',
    then the common parameter value from the main top level will be propagated down
    into the subsection dictionary and a list will be set as the value.  The list will 
    contain the toplevel value and the subsection value.

    Please run this module with '-e' as the argument to generate a sample .cfg file
    showing these various types of data.  And please see the self test function for 
    the various ways to call this function.

'''
    # set a local var name for ease of reading
    update = update_top_level_dict_from_section
    merge = merge_top_level_params_into_section

    if verbose: print(now(), 'parse_configuration_file: update_top_level_dict_from_section  =', update)
    if verbose: print(now(), 'parse_configuration_file: merge_top_level_params_into_section =', merge)

    # the main config dictionary to fill and return
    cfg_dict = {}

    # a list to hold the sub section names
    section_names = []

    # a temporary dictionary to hold subsection data 
    tmp_section_dict = {}

    # Get the initial variable to type dictionary for this cfg file 
    type_dict = parse_configuration_file_to_create_variable_type_relations( filename )

    # read the cfg file and double check for type
    in_file = open(filename,'r')

    # loop over lines and set up default entries in the type dict
    for line in in_file:

        # skip past comments and blank lines
        if line.startswith('#'):
            continue # to next line
        if line.rstrip() == '':
           continue # to next line

        # process section lines
        if '[' in line:

            # If currently working on a sub section dict
            if not tmp_section_dict == {} :
                # close out previous tmp_section_dict
                key = tmp_section_dict['_SECTION_NAME_']
                # add a deep copy do the master dict
                cfg_dict[key] = dict(tmp_section_dict)

                # clean out the tmp
                tmp_section_dict = {}

            # start up a new sub section dict 
            line = line.lstrip()
            line = line.rstrip()
            line = line.strip('[')
            line = line.strip(']')
            tmp_section_dict['_SECTION_NAME_'] = line
            section_names.append(line)

            continue # to next line

        # parse data line into variable, value and comment:
        var, val = line.split('=')

        # parse value from comment 
        com = ''
        if ';' in val:
            val, com = val.split(';')

        # clean up white space
        var = var.strip() 
        var = var.rstrip() 
       
        val = val.strip() 
        val = val.rstrip() 

        com = com.strip() 
        com = com.rstrip() 

        # now check for type, and correct entries
        type = ''

        try:
            type = type_dict[var][0]
        except KeyError:
            print('Variable', var, 'not found in type map')
            print(traceback.format_exc())
            sys.exit(1)

        if verbose: print( now(), 'parse_configuration_file: type = ', type)

        # check for values with no type; probably from a blank entry like: 'param = '
        if type == 'unk':
            print('Core_Util: parse_configuration_file: in file =', filename)
            print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value ="', val, '" but no value is set; type is not defined')
            sys.exit(1)

        # Variable to hold a typed value (bool, int, etc.)
        typed_value = None

        #
        # Single item data types
        #
        if type == 'bool':
            try:
                b = bool(val)
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as a bool failed')
                print(traceback.format_exc())
                sys.exit(1)

            # NOTE: cannot do assgiment directly from a call to bool()
            # >>> a = bool("False")
            # >>> a
            # True

            # Must set the values 'by hand' :
            if val == 'False' or val == 'false':
                typed_value = False

            if val == 'True' or val == 'true':
                typed_value = True

        elif type == 'int':
            try:
                i = int(val)
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as an int failed')
                print(traceback.format_exc())
                sys.exit(1)

            typed_value = i

        elif type == 'float':
            try:
                f = float(val)
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as an int failed')
                print(traceback.format_exc())
                sys.exit(1)

            typed_value = f

        elif type == 'string':
            try: 
                s = str(val)
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as an int failed')
                print(traceback.format_exc())
                sys.exit(1)

            typed_value = s

        elif type == 'path':
            try: 
                s = str(val)
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as an int failed')
                print(traceback.format_exc())
                sys.exit(1)

            typed_value = s

        # List data 
        elif type == 'int_list':
            try: 
                list = val.split(',')
                i = int(list[0])
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as an int list failed')
                print(traceback.format_exc())
                sys.exit(1)

            # remove white spaces around list items
            new_list = []
            for item in list:
                item = item.strip()
                item = item.rstrip()
                new_list.append( int(item) )
      
            typed_value = new_list
            
        elif type == 'float_list':
            try: 
                list = val.split(',')
                f = float(list[0])
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as a float list failed')
                print(traceback.format_exc())
                sys.exit(1)

            # remove white spaces around list items
            new_list = []
            for item in list:
                item = item.strip()
                item = item.rstrip()
                new_list.append( float(item) )
      
            typed_value = new_list

        elif type == 'string_list':
            try: 
                list = val.split(',')
                s = str(list[0])
            except ValueError:
                print('Core_Util: parse_configuration_file: in file =', filename)
                print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, 'as a string list failed')
                print(traceback.format_exc())
                sys.exit(1)

            # remove white spaces around list items
            new_list = []
            for item in list:
                item = item.strip()
                item = item.rstrip()
                new_list.append( item )
      
            typed_value = new_list

        else:
            print('Core_Util: parse_configuration_file: in file =', filename)
            print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value =', val, ' but no type defined')
            print(traceback.format_exc())
            sys.exit(1)

        if verbose: print('Core_Util: typed_value = ', typed_value)

        # Double check we have set a value
        if typed_value == None:
            print('Core_Util: parse_configuration_file: in file =', filename)
            print('Core_Util: parse_configuration_file: trying to read  "', var, '" with value ="', val, '" but no value was set.  Please check the .cfg file')
            sys.exit(1)


        # Some data type has been assigned or an excpetion thrown, now set the data 

        # Set the section data 
        if not tmp_section_dict == {} : 
            # check if var exists in top level dict 
            if not var in cfg_dict :
               # new var ; simply set the value in the section dictionary
               tmp_section_dict[var] = typed_value 
            else : 
               # var is in top level section dict ; check for merge 
               if merge: 
                   values_list = []
                   values_list.append( cfg_dict[var] ) # get the top level value
                   values_list.append( typed_value ) # get the current value 
                   tmp_section_dict[var] = flatten( values_list )
               else : 
                   # no merge ; simply set the value 
                   tmp_section_dict[var] = typed_value 

        # Set top level dict data 
        if var in cfg_dict: 
            # existing value ; check for update
            if update : cfg_dict[var] = typed_value 
        else: 
            # new value simply set it
            cfg_dict[var] = typed_value 

        continue # to next line

    # end of loop over lines

    # If currently working on a sub section dict
    if not tmp_section_dict == {} :
        # close out previous tmp_section_dict
        key = tmp_section_dict['_SECTION_NAME_']
        # add a deep copy do the master dict
        cfg_dict[key] = dict(tmp_section_dict)
        # clean out the tmp
        tmp_section_dict = {}

    # Add an entry to the top level dict with the sub section names
    cfg_dict['_SECTIONS_'] = section_names

    if verbose: print(now(), 'parse_configuration_file:')
    if verbose: tree_print(cfg_dict)

    return cfg_dict
#=====================================================================
#=====================================================================
def print_cfg_type_dict( d ):
   ''' print out a cfg type dictionary '''
   keys = sorted(d)
   for k in keys:
       print("variable ", ('%(k)-30s' % vars()), " is of type:", '%-10s' % d[k][0], "; sample value:", '%-20s' % d[k][1], "; comment:", d[k][2])

#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# Sample.cfg to test Core_Util.py like this: 
# $ ./Core_Util.py -e > _cu.cfg; ./Core_Util.py _cu.cfg

# Configuration files may have blank lines, comment lines and data lines.
# Blank lines and lines starting with # are ignored.  
# Data lines are processed into key/value pairs of a dictionary.

# Data lines are of the form:
# parameter_name = value ; comment

# Paramter names may be any string. 
# Values may be strings, ints, floats, booleans, paths,
# or comma separated lists of those base types.

# Comments may be any string but are not passed into the final dictionary

# Single value examples:
param_T = True ; boolean value 
param_F = False ; boolean value 
param_S = string_value  ; string value 
param_1 = 1 ; positive int
param_2 = -2 ; negative int
param_3 = 5.6 ; positive float
param_4 = -7.8 ; negative float
param_5 = 1.2e+3 ; positive float
param_6 = -1.2e-3 ; negative float, neg exponent
param_7 = 2.0e6 ; positive float 
param_8 = -2.0e6 ; negative float 
param_9 = 2e+06 ; etc.
param_10 = -2e+06
param_11 = 2e-06
param_12 = -2e-06

param_path = /usr/share/X11/rgb.txt ; path

# list value examples: NOTE: white space around list items will be removed in processing:
param_l0 = a, b,c ,d ; string list
param_l2 = 1,2 , 3 ; int list
param_l3 = 4.5,6.7 ; float list
param_l4 = -1,2,-3 ; int list
param_l3 = -4.5, 6.7 ; float list
param_l6 = 2e+06,  1.0e+5, -1.0e+5 ; float list

# Parameter names may be common to the top level and to subsections.
# Please see the documentation on the function 'parse_configuration_file()'
# for options on how to update the top level params from subsections, 
# and / or merge top level params into subsections.  
#
# By default top level params will not be updated, 
# and top level parms will not be merged into subsections

# top level common params examples:
common_int = 0
common_bool = True
common_str = value_from_top_level
common_int_list = 1,2,3
#
# Subsection names must be unique strings enclosed in [ ]
[Section_A]
param_a1 = 1.0
common_int = 1
common_int_list = 4,5,6
common_str = value_from_Section_A
#
[Section_B]
param_b1 = 100
common_int = 2
common_bool = False
common_str = value_from_Section_B
common_int_list = 7,8,9
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
def write_dictionary_to_file( arg, filename ):

    '''Sort dictionary keys in alphabetical order and write out
    key: value pairs to a file.'''

    if verbose: print( now(), 'write_dictionary_to_file:')

    log_file = open( filename, 'w' )
    log_file.write( now() + '\n' )

    keys = sorted( arg )
    for key in keys:
        line = '%s: %s\n' % (key, arg[key])
        log_file.write( line )

    log_file.close()

#=====================================================================
#=====================================================================
#=====================================================================
def write_cfg_dictionary( input_d, filename = False, write_top_level_data = True) :
    '''Process a geodyanmic frameworkd style dictionary, ard_d, and write it to standar out, or optional file on disk'''
    if verbose: print( now(), 'write_cfg_dictionary: input_d =')
    if verbose: tree_print( input_d )

    # make a deep copy of the dict since we modify things here
    out_d = copy.deepcopy( input_d ) 

    # strings to write out 
    out_str = ''
    sec_str = ''

    # process any sections first to remove them from out_d
    if out_d.get( '_SECTIONS_' ) :
        section_names = out_d['_SECTIONS_']
        for name in section_names :
            if verbose: print( now(), 'write_cfg_dictionary: -------- name =', name)
            sec_str += '[%(name)s]\n' % vars() 
            s = out_d[name]
            del s['_SECTION_NAME_'] # remove special key 
            keys = sorted( s ) 
            for k in keys :
                if type( s[k] ) is list:
                    sec_str += '%s = ' % k
                    for item in s[k][:-1] :
                        sec_str += '%s,' % str(item) 
                    sec_str += '%s\n' % s[k][-1]
                else : 
                    sec_str += '%s = %s\n' % (k, s[k])
            sec_str += '\n'
            # delete this section from the main dict 
            del out_d[name]
    # end of check on sections
    # delete section list from the main top level dict 
    del out_d['_SECTIONS_']

    # check for write pf top level data 
    if write_top_level_data :

        # get all the top level keys
        keys = sorted( out_d )
        for k in keys : 

            if type( out_d[k] ) is list:
                out_str += '%s = ' % k
                for item in out_d[k][:-1] :
                    out_str += '%s,' % str(item) 
                out_str += '%s\n' % out_d[k][-1]
            else : 
                out_str += '%s = %s\n' % (k, out_d[k])

    # if needed append the sections string
    if not sec_str == '' : out_str += sec_str 

    if verbose: print(now(), 'write_cfg_dictionary: out_str =')
    if verbose: print(out_str)

    # either write to a file or std out 
    if filename : 
       file = open( filename, 'w' )
       file.write( out_str )
       file.close()
    else : 
       print(out_str)
#=====================================================================
#=====================================================================
#=====================================================================
def get_spec_dictionary(spec_string) :
    '''Process a multi-value spec string, and return a new dictionary with these lists:

    spec_d['spec_string'] = the original spec as a string
    spec_d['list'] = the list of specific values from processing the spec

A multi value spec string may be given as: 

    a single item, without units 
        e.g.: '1000', or '250', or '100'

    a comma delimited list of values, without units 
        e.g.: '1000,100,10'

    a slash delimited start/end/step set, without units 
        e.g.: '250/25/10'

    or a filename with indidual single value entries, without units, one item per line.'''

    # convert the input to a string
    spec = str(spec_string)

    if verbose: print( now(), 'get_spec_dictionary: spec =', spec )

    # spec dictionary to build and return
    spec_d = { 'spec': spec, 'list': [] } 

    # the list of times from parsing a spec string
    list = [] 

    # check for discrete times separated via comma 
    if spec.count(','):
        spec = spec.replace('[', '')
        spec = spec.replace(']', '')
        list = spec.split(',')
        list = [ i.strip() for i in list ] 
        list = [ i.rstrip() for i in list ]

    # check for sequence of values separated via / 
    elif spec.count('/'): 
        if len( spec.split('/') ) != 3 :
            msg = "Three numbers are required: start/end/step"
            raise ValueError(msg)

        (start,end,step) = spec.split('/') 

        units = ''
        start = int(start)
        end = int(end)
        step = int(step)

        # build list, add back units
        if (start < end):
            t = start
            while t <= end:
                list.append(str(t) + units)
                t = t + step
        elif (start > end):
            t = start
            while t >= end:
                list.append(str(t) + units)
                t = t - step

    # check for data file 
    elif spec.endswith('.dat'):
        list = Core_Util.parse_general_one_value_per_line_type_file(spec)

    # else, single time spec
    else :
        list = [ spec ]

    if verbose: print( now(), 'get_spec_dictionary:', list )

    # update list
    spec_d['list'] = list

    return spec_d

#=====================================================================
#=====================================================================
def clean_case_of_grid_maker_files(a):
    '''clean up all the output of a grid_maker run; USE WITH CAUTION'''
	
    cmd = ''
    if a == 'all': 
        cmd = 'rm -v *.cpt *.grd *.png *.ps *.xyz'
    else :
        cmd = 'rm -v *.cpt *.png *.ps *.xyz'

    print( now(), cmd )
    subprocess.call( cmd, shell=True )

#=====================================================================
#=====================================================================
def test():
    '''geodynamic framework self test'''

    global verbose
    verbose = True

    print(now(), 'Core_Util.test(): sys.argv = ', sys.argv )

    # read the defaults
    frame_d = parse_geodynamic_framework_defaults()

    # Read the first argument as a .cfg file , no update , no merge
    print('')
    print('--------------------------------------------------------------------------')
    print(now(), 'Core_Util.test: parse_configuration_file(cfg)')
    cfg_d = parse_configuration_file( sys.argv[2] )
    tree_print(cfg_d)

    # Read the first argument as a .cfg file , no update , yes merge
    print('')
    print('--------------------------------------------------------------------------')
    print(now(), 'Core_Util.test: parse_configuration_file(cfg, update=False, merge=True):')
    cfg_d = parse_configuration_file( sys.argv[2], False, True )
    tree_print(cfg_d)

    # Read the first argument as a .cfg file , yes update , no merge
    print('')
    print('--------------------------------------------------------------------------')
    print(now(), 'Core_Util.test: parse_configuration_file(cfg, update=True, merge=False):')
    cfg_d = parse_configuration_file( sys.argv[2], True, False )
    tree_print(cfg_d)

    # Read the first argument as a .cfg file , yes update , yes merge
    print('')
    print('--------------------------------------------------------------------------')
    print(now(), 'Core_Util.test: parse_configuration_file(cfg, update=True, merge=True):')
    cfg_d = parse_configuration_file( sys.argv[2], True, True )
    tree_print(cfg_d)

    # test writing out the dictionary
    #write_cfg_dictionary( cfg_d, './_TEST_from_write_cfg_dictionary.cfg' )

    # ... Additional self-tests go here ...

    ## test the parse spec
    #if 'spec_test' in cfg_d :
    #    spec_d = get_spec_dictionary( cfg_d['spec_test'] )
    #    print( spec_d ) 

#=====================================================================
#=====================================================================
def rename_function() :
    '''quick and easy way to copy or rename a batch of files'''

    file_list = sys.argv[2:]

    for f in file_list:

        # Please NOTE: adjust these parsing / replace lines to match the patterns

        # parse the format from the last entry in the file name
        format = f.split('.')[-1]

        # parse the age out of the remaining string 
        age = f.replace('.' + format, '')
        age = age.replace('sT', '')
        # zero pad the age
        a = '%03d' % int( age )

        # build the new name
        #n = 'input-dynamic_topography-' + str(a) + 'Ma-0km' + '.' + format
        n = 'input-total_topography-' + str(a) + 'Ma-0km' + '.' + format
        print( now(), 'rename f=', f, 'to n=', n)

        # create the shell command
        cmd = 'cp -v %(f)s %(n)s' % vars()
        print( now(), cmd )
        subprocess.call( cmd, shell=True )

#=====================================================================
def test_function( args ):
    '''test specific functions'''
    global verbose 
    verbose = True 

    # NOTE: comment or uncomment the specfic functions to test here:

    file = args[2]
    print( now(), 'test_function: file =', file)

    #sz_dict = get_subduction_zone_data( file )

    sz_dict = get_subduction_systems( file )

    print( now(), 'sz_dict =')
    tree_print(sz_dict)

#=====================================================================
#=====================================================================
#=====================================================================
if __name__ == "__main__":
    import Core_Util

    if len( sys.argv ) > 1:

        # make the example configuration file 
        if sys.argv[1] == '-e':
            make_example_config_file()
            sys.exit(0)

        # run a specific test on sys.argv
        if sys.argv[1] == '-t':
            #test_function( sys.argv )
            test()
            sys.exit(0)

        # run batch rename function
        if sys.argv[1] == '-r':
            rename_function()
            sys.exit(0)

        # run a clean up process; USE WITH CAUTION
        if sys.argv[1] == '-c':
            clean_case_of_grid_maker_files('')
            sys.exit(0)

        # run a clean up ALL process; USE WITH CAUTION
        if sys.argv[1] == '-ca':
            clean_case_of_grid_maker_files('all')
            sys.exit(0)

        # process sys.arv as file names for testing 
        test( sys.argv )
    else:
        # print module documentation and exit
        help(Core_Util)

#=====================================================================
#=====================================================================
