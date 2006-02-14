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

#if !defined(journal_FacilityMap_h)
#define journal_FacilityMap_h


// forward declarations
namespace journal {
    class Index;
    class FacilityMap;
}


class journal::FacilityMap : public journal::Index {

// types
public:
    typedef std::map<string_t, state_t *> index_t;
    typedef std::pair<string_t, state_t *> entry_t;

// interface
public:
    virtual state_t & lookup(string_t name);

// meta-methods
public:
    virtual ~FacilityMap();
    inline FacilityMap(bool defaultState = true);

// disable these
private:
    FacilityMap(const FacilityMap &);
    const FacilityMap & operator=(const FacilityMap &);

// data
private:
    index_t _index;
    bool _defaultState;
};

// get the inline definitions
#define journal_FacilityMap_icc
#include "FacilityMap.icc"
#undef journal_FacilityMap_icc

#endif
// version
// $Id: FacilityMap.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
