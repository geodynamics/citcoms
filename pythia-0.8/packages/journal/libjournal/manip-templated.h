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

#if !defined(journal_manip_templated_h)
#define journal_manip_templated_h

namespace journal {

    // manipulators with one argument
    template <typename arg1_t> class manip_1;

    // manipulators with two arguments
    template <typename arg1_t, typename arg2_t> class manip_2;

    // manipulators with three arguments
    template <typename arg1_t, typename arg2_t, typename arg3_t> class manip_3;

    // typedefs of the builtin manipulators
    typedef manip_2<const char *, long> loc2_t;
    typedef manip_2<const char *, const char *> set_t;
    typedef manip_3<const char *, long, const char *> loc3_t;
}

// the injection operators: leave these in the global namespace

template <typename arg1_t>
journal::Diagnostic & operator << (
    journal::Diagnostic &, journal::manip_1<arg1_t>);

template <typename arg1_t, typename arg2_t>
journal::Diagnostic & operator << (
    journal::Diagnostic &, journal::manip_2<arg1_t, arg2_t>);

template <typename arg1_t, typename arg2_t, typename arg3_t>
journal::Diagnostic & operator << (
    journal::Diagnostic &, journal::manip_3<arg1_t, arg2_t, arg3_t>);


template <typename arg1_t>
class journal::manip_1 {

    friend journal::Diagnostic &
    ::operator<< <> (journal::Diagnostic &, journal::manip_1<arg1_t>);

// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, arg1_t);

// meta-methods
public:

    manip_1(factory_t f, arg1_t arg1);

// data
private:

    factory_t _f;
    arg1_t _arg1;
};


template <typename arg1_t, typename arg2_t>
class journal::manip_2 {

    friend journal::Diagnostic & 
    ::operator<< <> (journal::Diagnostic &, journal::manip_2<arg1_t, arg2_t>);

// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, arg1_t, arg2_t);

// meta-methods
public:

    manip_2(factory_t f, arg1_t arg1, arg2_t arg2);

// data
private:

    factory_t _f;
    arg1_t _arg1;
    arg2_t _arg2;
};


template <typename arg1_t, typename arg2_t, typename arg3_t>
class journal::manip_3 {

    friend journal::Diagnostic & 
    ::operator<< <> (journal::Diagnostic &, journal::manip_3<arg1_t, arg2_t, arg3_t>);

// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, arg1_t, arg2_t, arg3_t);

// meta-methods
public:

    manip_3(factory_t f, arg1_t arg1, arg2_t arg2, arg3_t arg3);

// data
private:

    factory_t _f;
    arg1_t _arg1;
    arg2_t _arg2;
    arg3_t _arg3;
};

// get the inlined definitions
#define journal_manip_templated_icc
#include "manip-templated.icc"
#undef journal_manip_templated_icc

#endif

// version
// $Id: manip-templated.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file
