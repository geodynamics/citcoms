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

#if !defined(journal_Journal_h)
#define journal_Journal_h

// forward declarations
namespace journal {
    class Entry;
    class Device;
    class Journal;
}

// 
class journal::Journal
{
// types
public:
    typedef Entry entry_t;
    typedef Device device_t;

// interface
public:
    
    inline void device(device_t * newDevice);
    inline void record(const entry_t & entry);

// meta-methods
public:
    Journal();
    virtual ~Journal();

// disable these
private:
    Journal(const Journal &);
    const Journal & operator=(const Journal &);

// data
private:
    device_t * _device;
};

// get the inline definitions
#define journal_Journal_icc
#include "Journal.icc"
#undef journal_Journal_icc

#endif

// version
// $Id: Journal.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file
