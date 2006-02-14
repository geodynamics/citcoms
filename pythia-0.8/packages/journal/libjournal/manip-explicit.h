// -*- C++ -*-
//
//--------------------------------------------------------------------------------
//
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
//--------------------------------------------------------------------------------
//

#if !defined(journal_manip_explicit_h)
#define journal_manip_explicit_h

namespace journal {

    // declarations of the builtin manipulators
    class set_t;
    class loc2_t;
    class loc3_t;
}

// the injection operators: leave these in the global namespace
inline journal::Diagnostic & operator << (journal::Diagnostic &, journal::set_t);
inline journal::Diagnostic & operator << (journal::Diagnostic &, journal::loc2_t);
inline journal::Diagnostic & operator << (journal::Diagnostic &, journal::loc3_t);

class journal::set_t {
// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, const char *, const char *);

// meta-methods
public:
    set_t(factory_t, const char *, const char *);

// data
public:
    factory_t _f;
    const char * _key;
    const char * _value;
};


class journal::loc2_t {

// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, const char *, long);

// meta-methods
public:
    loc2_t(factory_t, const char *, long);

// data
public:
    factory_t _f;
    const char * _file;
    long _line;
};


class journal::loc3_t {
// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, const char *, long, const char *);

// meta-methods
public:
    loc3_t(factory_t, const char *, long, const char *);

// data
public:
    factory_t _f;
    const char * _file;
    long _line;
    const char * _function;
};

// get the inline definitions
#define journal_manip_explicit_icc
#include "manip-explicit.icc"
#undef journal_manip_explicit_icc

#endif

// version
// $Id: manip-explicit.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file
