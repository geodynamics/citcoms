// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_Convertor_h)
#define pyCitcom_Convertor_h

#include <memory>
#include "Array2D.h"
#include "BoundedBox.h"
#include "DIM.h"

struct All_variables;
class CartesianCoord;
class SIUnit;


// singleton class

class Convertor {
    static std::auto_ptr<SIUnit> si;
    static std::auto_ptr<CartesianCoord> cart;
    static bool inited;

public:
    ~Convertor() {};

    static void init(bool dimensional, bool transformational,
		     const All_variables* E);
    static Convertor& instance();  // the singleton

    // internal representation ==> standard representation
    void coordinate(BoundedBox& bbox) const;
    void coordinate(Array2D<double,DIM>& X) const;
    void temperature(Array2D<double,1>& T) const;
    void time(double& t) const;
    void traction(Array2D<double,DIM>& F,
		  const Array2D<double,DIM>& X) const;
    void velocity(Array2D<double,DIM>& V,
		  const Array2D<double,DIM>& X) const;

    // standard representation ==> internal representation
    void xcoordinate(BoundedBox& bbox) const;
    void xcoordinate(Array2D<double,DIM>& X) const;
    void xtemperature(Array2D<double,1>& T) const;
    void xtime(double& t) const;
    void xtraction(Array2D<double,DIM>& F,
		   const Array2D<double,DIM>& X) const;
    void xvelocity(Array2D<double,DIM>& V,
		   const Array2D<double,DIM>& X) const;

private:
    Convertor();

    // disable
    Convertor(const Convertor&);
    Convertor& operator=(const Convertor&);

};


#endif

// version
// $Id: Convertor.h,v 1.1 2004/01/07 21:54:00 tan2 Exp $

// End of file
