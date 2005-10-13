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

#if !defined(journalmodule_ProxyDevice_h)
#define journalmodule_ProxyDevice_h

// 
class ProxyDevice : public journal::Device
{
// interface
public:
    virtual void record(const entry_t &);

// meta-methods
public:
    virtual ~ProxyDevice();
    inline ProxyDevice(PyObject * journal);

// data
private:
    PyObject * _journal;

// disable these
private:
    ProxyDevice(const ProxyDevice &);
    const ProxyDevice & operator=(const ProxyDevice &);
};

#endif

// get the inline definitions
#include "ProxyDevice.icc"

// version
// $Id: ProxyDevice.h,v 1.1.1.1 2005/03/08 16:13:54 aivazis Exp $

// End of file
