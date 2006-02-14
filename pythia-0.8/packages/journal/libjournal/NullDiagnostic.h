// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             Michael A.G. Aivazis
//                      California Institute of Technology
//                      (C) 1998-2005  All Rights Reserved
//
// {LicenseText}
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(journal_NullDiagnostic_h)
#define journal_NullDiagnostic_h


// forward declarations
namespace journal {
    class NullDiagnostic;
}

// injection operator
template <typename item_t>
inline journal::NullDiagnostic & operator<< (journal::NullDiagnostic &, item_t);


class journal::NullDiagnostic {

// types
public:
    typedef std::string string_t;

// interface
public:
    inline void state(bool) const { return; }
    inline bool state() const { return false; }

    inline void activate() const { return; }
    inline void deactvate() const { return; }

// meta-methods
    ~NullDiagnostic() {}
    NullDiagnostic(string_t) {}

// disable these
private:
    NullDiagnostic(const NullDiagnostic &);
    const NullDiagnostic & operator=(const NullDiagnostic &);
};

// manipulators
namespace journal {
    inline const NullDiagnostic & endl(const NullDiagnostic &);
    inline const NullDiagnostic & newline(const NullDiagnostic &);
}


// get the inline definitions
#define journal_NullDiagnostic_icc
#include "NullDiagnostic.icc"
#undef journal_NullDiagnostic_icc

# endif

// version
// $Id: NullDiagnostic.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
