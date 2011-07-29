/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */


#include <vector>
#include <list>
#include <math.h>

/* forward declaration */
struct All_variables;

#ifndef _TRACER_DEFS_H_
#define _TRACER_DEFS_H_

typedef int ElementID;
#define UNDEFINED_ELEMENT		((ElementID)-99)

class CartesianCoord;

// Position or vector in spherical coordinates
class SphericalCoord {
public:
	double	_theta, _phi, _rad;
	SphericalCoord(void) : _theta(0), _phi(0), _rad(0) {};
	SphericalCoord(double theta, double phi, double rad) : _theta(theta), _phi(phi), _rad(rad) {};
	
	size_t size(void) const { return 3; };
	double *writeToMem(double *mem) const;
	double *readFromMem(double *mem);
	CartesianCoord toCartesian(void) const;
	
	void constrainThetaPhi(void);
	double constrainAngle(const double angle) const;
};

// Position or vector in Cartesian coordinates
class CartesianCoord {
public:
	double	_x, _y, _z;
	CartesianCoord(void) : _x(0), _y(0), _z(0) {};
	CartesianCoord(double x, double y, double z) : _x(x), _y(y), _z(z) {};
	
	size_t size(void) const { return 3; };
	double *writeToMem(double *mem) const;
	double *readFromMem(double *mem);
	SphericalCoord toSpherical(void) const;
	CartesianCoord crossProduct(const CartesianCoord &b) const;
	double dist(const CartesianCoord &o) const {
		double xd=_x-o._x, yd=_y-o._y, zd=_z-o._z;
		return sqrt(xd*xd+yd*yd+zd*zd);
	};
	
	const CartesianCoord operator+(const CartesianCoord &other) const;
	const CartesianCoord operator-(const CartesianCoord &other) const;
	const CartesianCoord operator*(const double &val) const;
	const CartesianCoord operator/(const double &val) const;
	void operator=(const CartesianCoord &other) { _x = other._x; _y = other._y; _z = other._z; };
};

class CapBoundary {
public:
	CartesianCoord	cartesian_boundary[4];
	SphericalCoord	spherical_boundary[4];
	double			cos_theta[4];
	double			sin_theta[4];
	double			cos_phi[4];
	double			sin_phi[4];
	
	void setBoundary(int bnum, SphericalCoord sc);
	void setCartTrigBounds(int bnum, CartesianCoord cc, double cost, double sint, double cosf, double sinf);
};

class Tracer {
private:
	// Tracer position in spherical coordinates
	SphericalCoord	_sc;
	// Tracer position in Cartesian coordinates
	CartesianCoord	_cc;
	// Previous Cartesian position
	CartesianCoord	_cc0;
	// Previous Cartesian velocity
	CartesianCoord	_Vc;
	
	// Tracer flavor (meaning should be application dependent)
	double _flavor;
	
	// ID of element containing this tracer
	ElementID _ielement;
	
public:
	Tracer(void) : _sc(), _cc(), _cc0(), _Vc(), _flavor(0), _ielement(UNDEFINED_ELEMENT) {};
	Tracer(SphericalCoord new_sc, CartesianCoord new_cc) :
		_sc(new_sc), _cc(new_cc), _cc0(), _Vc(), _flavor(0), _ielement(UNDEFINED_ELEMENT) {};
	
	CartesianCoord getCartesianPos(void) const { return _cc; };
	SphericalCoord getSphericalPos(void) const { return _sc; };
	CartesianCoord getOrigCartesianPos(void) const { return _cc0; };
	CartesianCoord getCartesianVel(void) const { return _Vc; };
	
	void setCoords(const SphericalCoord new_sc, const CartesianCoord new_cc) {
		_sc = new_sc;
		_cc = new_cc;
	}
	void setOrigVals(const CartesianCoord new_cc0, const CartesianCoord new_vc) {
		_cc0 = new_cc0;
		_Vc = new_vc;
	}
	
	double theta(void) { return _sc._theta; };
	double phi(void) { return _sc._phi; };
	double rad(void) { return _sc._rad; };
	
	double x(void) { return _cc._x; };
	double y(void) { return _cc._y; };
	double z(void) { return _cc._z; };
	
	ElementID ielement(void) const { return _ielement; };
	void set_ielement(const ElementID ielement) { _ielement = ielement; };
	
	double flavor(void) const { return _flavor; };
	void set_flavor(const double flavor) { _flavor = flavor; };
	
	size_t size(void);
	double *writeToMem(double *mem) const;
	double *readFromMem(double *mem);
};

typedef std::list<Tracer> TracerList;

#endif

struct TRACE{

    FILE *fpt;

    char tracer_file[200];

    int itracer_warnings;
    int ianalytical_tracer_test;
    int ic_method;
    int itperel;
    int itracer_interpolation_scheme;

    double box_cushion;

    /* tracer arrays */
	
	// Sets of tracers organized by cap
    TracerList *tracers;
	
	// Sets of tracers that have escaped a cap, organized by cap
    TracerList *escaped_tracers;

    /* tracer flavors */
    int nflavors;
    int **ntracer_flavor[13];

	int number_of_tracers;
	
    int ic_method_for_flavors;
    double *z_interface;

    char ggrd_file[255];		/* for grd input */
    int ggrd_layers;

    /* statistical parameters */
    int istat_ichoice[13][5];
    int istat_isend;
    int istat_iempty;
    int istat1;
    int istat_elements_checked;
    int ilast_tracer_count;

    /* timing information */
    double advection_time;
    double find_tracers_time;
    double lost_souls_time;

    /* Mesh information */
	CapBoundary	boundaries[13];

    /*********************/
    /* for global model  */
    /*********************/

    /* regular mesh parameters */
    int numtheta[13];
    int numphi[13];
    unsigned int numregel[13];
    unsigned int numregnodes[13];
    double deltheta[13];
    double delphi[13];
    double thetamax[13];
    double thetamin[13];
    double phimax[13];
    double phimin[13];
    int *regnodetoel[13];
    int *regtoel[13][5];

    /* gnomonic shape functions */
    double *shape_coefs[13][3][10];

    /**********************/
    /* for regional model */
    /**********************/

    double *x_space;
    double *y_space;
    double *z_space;

    /*********************/
    /* function pointers */
    /*********************/

    ElementID (* iget_element)(struct All_variables*, int, int,
                         CartesianCoord, SphericalCoord);

    CartesianCoord (* get_velocity)(struct All_variables*, int, int,
                          SphericalCoord);

    void (* keep_within_bounds)(struct All_variables*,
                                CartesianCoord &, SphericalCoord &);
};
