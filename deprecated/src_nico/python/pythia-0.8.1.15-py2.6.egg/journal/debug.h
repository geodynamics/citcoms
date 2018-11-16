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

#if !defined(journal_debug_h)
#define journal_debug_h

#include <string>

#if !defined(WITHOUT_JOURNAL_DEBUG)
#include <sstream>

#include "Diagnostic.h"
#include "SeverityDebug.h"

#include "macros.h"
#include "manipulators.h"

// forward declarations
namespace journal {

    typedef SeverityDebug debug_t;
}

#else

#include "NullDiagnostic.h"
// manipulators

// forward declarations
namespace journal {

    typedef NullDiagnostic debug_t;
}


#endif


#endif

// version
// $Id: debug.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
