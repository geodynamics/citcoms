// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Dimensional_h)
#define pyCitcom_Dimensional_h

#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"

struct All_variables;


// singleton class

class Dimensional {
    static const All_variables* E;

    const double length_factor;
    const double velocity_factor;
    const double temperature_factor;
    const double time_factor;
    const double traction_factor;

public:
    ~Dimensional() {};

    static void setE(const All_variables* E);
    static Dimensional& instance();   // the singleton

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
    Dimensional(const All_variables* E);

    // disable
    Dimensional();
    Dimensional(const Dimensional&);
    Dimensional& operator=(const Dimensional&);

};




#endif

// version
// $Id: Dimensional.h,v 1.1 2003/12/30 21:44:32 tan2 Exp $

// End of file
