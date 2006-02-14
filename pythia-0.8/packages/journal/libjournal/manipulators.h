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

#if !defined(journal_manipulators_h)
#define journal_manipulators_h

// get infrastructure manipulator definitions/declaration
#if defined(JOURNAL_NON_TEMPLATED_MANIPULATORS)
#include "manip-explicit.h"
#else
#include "manip-templated.h"
#endif

// forward declarations
namespace journal {
    class Diagnostic;

    inline Diagnostic & endl(Diagnostic &);
    inline Diagnostic & newline(Diagnostic &);

    inline set_t set(const char *, const char *);
    inline Diagnostic & __diagmanip_set(Diagnostic &, const char *, const char *);

    inline loc2_t at(const char *, long);
    inline Diagnostic & __diagmanip_loc(Diagnostic &, const char *, long);

    inline loc3_t at(const char *, long, const char *);
    inline Diagnostic & __diagmanip_loc(Diagnostic &, const char *, long, const char *);
}

inline journal::Diagnostic & 
operator<< (journal::Diagnostic &, journal::Diagnostic & (*)(journal::Diagnostic &));

// get the inline definitions
#define journal_manipulators_icc
#include "manipulators.icc"
#undef journal_manipulators_icc

#endif

// version
// $Id: manipulators.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
