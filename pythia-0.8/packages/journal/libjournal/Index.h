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

#if !defined(journal_Index_h)
#define journal_Index_h


// forward declarations
namespace journal {
    class Index;
    class Diagnostic;
}


class journal::Index {

// types
public:
    typedef std::string string_t;
    typedef Diagnostic::state_t state_t;

// interface
public:
    virtual state_t & lookup(string_t name) = 0;

// meta-methods
public:
    virtual ~Index();
    inline Index();

// disable these
private:
    Index(const Index &);
    const Index & operator=(const Index &);
};

// get the inline definitions
#define journal_Index_icc
#include "Index.icc"
#undef journal_Index_icc

#endif
// version
// $Id: Index.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
