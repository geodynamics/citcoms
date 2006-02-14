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

#if !defined(journal_SeverityFirewall_h)
#define journal_SeverityFirewall_h


// forward declarations
namespace journal {
    class Diagnostic;
    class SeverityFirewall;
    class Index;
}


class journal::SeverityFirewall : public journal::Diagnostic {

// types
public:
    typedef Index index_t;

// interface
public:
    inline string_t name() const;
    static state_t & lookup(string_t);

// meta-methods
public:
    virtual ~SeverityFirewall();
    inline SeverityFirewall(string_t);

// disable these
private:
    SeverityFirewall(const SeverityFirewall &);
    const SeverityFirewall & operator=(const SeverityFirewall &);

// data
private:
    static index_t * _index;
};

// get the inline definitions
#define journal_SeverityFirewall_icc
#include "SeverityFirewall.icc"
#undef journal_SeverityFirewall_icc

#endif
// version
// $Id: SeverityFirewall.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
