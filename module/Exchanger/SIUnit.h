// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_SIUnit_h)
#define pyCitcom_SIUnit_h

#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"

struct All_variables;


// singleton class

class SIUnit {
    const double length_factor;
    const double velocity_factor;
    const double temperature_factor;
    const double time_factor;
    const double traction_factor;

public:
    SIUnit(const All_variables* E);
    ~SIUnit() {};

    // dimensionalize
    void coordinate(BoundedBox& bbox) const;
    void coordinate(Array2D<double,DIM>& X) const;
    void temperature(Array2D<double,1>& T) const;
    void time(double& t) const;
    void traction(Array2D<double,DIM>& F) const;
    void velocity(Array2D<double,DIM>& V) const;

    // non-dimensionalize
    void xcoordinate(BoundedBox& bbox) const;
    void xcoordinate(Array2D<double,DIM>& X) const;
    void xtemperature(Array2D<double,1>& T) const;
    void xtime(double& t) const;
    void xtraction(Array2D<double,DIM>& F) const;
    void xvelocity(Array2D<double,DIM>& V) const;

private:
    // disable
    SIUnit(const SIUnit&);
    SIUnit& operator=(const SIUnit&);

};




#endif

// version
// $Id: SIUnit.h,v 1.2 2004/01/07 21:54:00 tan2 Exp $

// End of file
