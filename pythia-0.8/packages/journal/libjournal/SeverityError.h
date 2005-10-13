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

#if !defined(journal_SeverityError_h)
#define journal_SeverityError_h


// forward declarations
namespace journal {
    class Diagnostic;
    class SeverityError;
    class Index;
}


class journal::SeverityError : public journal::Diagnostic {

// types
public:
    typedef Index index_t;

// interface
public:
    inline string_t name() const;
    static state_t & lookup(string_t);

// meta-methods
public:
    virtual ~SeverityError();
    inline SeverityError(string_t);

// disable these
private:
    SeverityError(const SeverityError &);
    const SeverityError & operator=(const SeverityError &);

// data
private:
    static index_t * _index;
};

// get the inline definitions
#define journal_SeverityError_icc
#include "SeverityError.icc"
#undef journal_SeverityError_icc

#endif
// version
// $Id: SeverityError.h,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file 
