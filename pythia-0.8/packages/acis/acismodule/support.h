// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyacis_support_h)
#define pyacis_support_h

class outcome;
void throwACISError(const outcome &, const char *, PyObject *);

class ACISModeler {

public:

    static bool initialize();
};

#endif

// version
// $Id: support.h,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
