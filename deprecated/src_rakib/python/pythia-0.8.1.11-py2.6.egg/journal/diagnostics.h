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

#if !defined(journal_info_h)
#define journal_info_h

#include <string>
#include <sstream>

#include "Diagnostic.h"

#include "SeverityDebug.h"
#include "SeverityError.h"
#include "SeverityFirewall.h"
#include "SeverityInfo.h"
#include "SeverityWarning.h"

#include "macros.h"
#include "manipulators.h"

// forward declarations
namespace journal {

    typedef SeverityError error_t;
    typedef SeverityInfo info_t;
    typedef SeverityWarning warning_t;

    typedef SeverityDebug debug_t;
    typedef SeverityFirewall firewall_t;
}


#endif

// version
// $Id: diagnostics.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
