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

#if !defined(journal_SeverityInfo_h)
#define journal_SeverityInfo_h


// forward declarations
namespace journal {
    class Diagnostic;
    class SeverityInfo;
}


class journal::SeverityInfo : public journal::Diagnostic {

// interface
public:
    string_t name() const { return  "info." + facility(); }

// meta-methods
public:
    ~SeverityInfo() {}
    
    SeverityInfo(string_t name) :
        Diagnostic(name, "info") {}

// disable these
private:
    SeverityInfo(const SeverityInfo &);
    const SeverityInfo & operator=(const SeverityInfo &);
};


#endif
// version
// $Id: SeverityInfo.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
