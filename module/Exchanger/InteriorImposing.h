// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyCitcom_InteriorImposing_h)
#define pyCitcom_InteriorImposing_h

#include "Array2D.h"
#include "DIM.h"

struct All_variables;
class Interior;
class Sink;
class Source;


class InteriorImposingSink {
    All_variables* E;
    const Interior& interior;
    const Sink& sink;
    Array2D<double,1> tic;

public:
    InteriorImposingSink(const Interior& interior, const Sink& sink,
			  All_variables* E);
    ~InteriorImposingSink() {};

    void recvT();
    void imposeIC();

private:
    void imposeTIC();

};


class InteriorImposingSource {
    All_variables* E;
    const Source& source;
    Array2D<double,1> tic;

public:
    InteriorImposingSource(const Source& src, All_variables* E);
    ~InteriorImposingSource() {};

    void sendT();

private:

};


#endif

// version
// $Id: InteriorImposing.h,v 1.2 2003/11/07 21:43:47 puru Exp $

// End of file
