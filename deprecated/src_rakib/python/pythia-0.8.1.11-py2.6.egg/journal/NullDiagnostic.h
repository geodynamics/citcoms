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


class journal::NullDiagnostic {

// types
public:
    typedef std::string string_t;

// interface
public:
    void state(bool) const { return; }
    bool state() const { return false; }

    void activate() const { return; }
    void deactvate() const { return; }

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
    
    const NullDiagnostic & endl(const NullDiagnostic & diagnostic) {
        return diagnostic;
    }
    
    const NullDiagnostic & newline(const NullDiagnostic & diagnostic) {
        return diagnostic;
    }
    
}


// the injection operator
template <typename item_t>
inline journal::NullDiagnostic & operator<< (journal::NullDiagnostic & diagnostic, item_t) {
    return diagnostic;
}


#endif

// version
// $Id: NullDiagnostic.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
