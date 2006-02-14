// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             Michael A.G. Aivazis
//                      California Institute of Technology
//                      (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(journal_Facility_h)
#define journal_Facility_h


// forward declarations
namespace journal {
    class Facility;
}


class journal::Facility {
//interface
public:
    inline void state(bool);
    inline bool state() const;

// meta-methods
public:
    inline ~Facility();
    inline Facility(bool state);

// disable these
private:
    Facility(const Facility &);
    const Facility & operator=(const Facility &);

// data
private:
    bool _state;
};

// get the inline definitions
#define journal_Facility_icc
#include "Facility.icc"
#undef journal_Facility_icc

#endif
// version
// $Id: Facility.h,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file 
