// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcomSExchanger_Convertor_h)
#define pyCitcomSExchanger_Convertor_h

#include "Exchanger/Convertor.h"

struct All_variables;
class CartesianCoord;
class SIUnit;


// singleton class

class Convertor : public Exchanger::Convertor {

public:
    ~Convertor();

    static void init(bool dimensional, bool transformational,
		     const All_variables* E);

private:
    Convertor();

    // disable
    Convertor(const Convertor&);
    Convertor& operator=(const Convertor&);

};


#endif

// version
// $Id: Convertor.h,v 1.3 2004/05/11 07:55:30 tan2 Exp $

// End of file
