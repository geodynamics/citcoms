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

#if !defined(journal_SeverityDebug_h)
#define journal_SeverityDebug_h


// forward declarations
namespace journal {
    class Diagnostic;
    class SeverityDebug;
    class Index;
}


class journal::SeverityDebug : public journal::Diagnostic {

// types
public:
    typedef Index index_t;

// interface
public:
    inline string_t name() const;
    static state_t & lookup(string_t);

// meta-methods
public:
    virtual ~SeverityDebug();
    inline SeverityDebug(string_t);

// disable these
private:
    SeverityDebug(const SeverityDebug &);
    const SeverityDebug & operator=(const SeverityDebug &);

// data
private:
    static index_t * _index;
};

// get the inline definitions
#define journal_SeverityDebug_icc
#include "SeverityDebug.icc"
#undef journal_SeverityDebug_icc

#endif
// version
// $Id: SeverityDebug.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
