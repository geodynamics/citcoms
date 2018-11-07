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
}


class journal::SeverityFirewall : public journal::Diagnostic {

// interface
public:
    string_t name() const { return  "firewall." + facility(); }

// meta-methods
public:
    ~SeverityFirewall() {}
    
    SeverityFirewall(string_t name) :
        Diagnostic(name, "firewall") {}

// disable these
private:
    SeverityFirewall(const SeverityFirewall &);
    const SeverityFirewall & operator=(const SeverityFirewall &);
};


#endif
// version
// $Id: SeverityFirewall.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
