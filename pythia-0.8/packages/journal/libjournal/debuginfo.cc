// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                              Michael A.G. Aivazis
//                       California Institute of Technology
//                       (C) 1998-2003  All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//


#if defined(HAVE_CONFIG_H)
#include <config.h>
#elif defined(BLD_PROCEDURE)
#include <portinfo>
#endif

#include <cstdio>
#include <cstdarg>
#include <string>

#include "debuginfo.h"
#include "debug.h"


extern "C"
void debuginfo_out(const char * category, __HERE_DECL__, const char * fmt, ...)
{
    // create temporary Debug object and log message
    std::string catname(category);
    journal::debug_t dbg(catname);

    if (debuginfo_active(category)) {
        std::va_list args;
        char buffer[2048];

	va_start(args, fmt);
        std::vsprintf(buffer, fmt, args);
        va_end(args);

        dbg << journal::at(__HERE_ARGS__)
            << buffer << journal::endl;
    }

    return;
}

extern "C"
int debuginfo_active(const char * category)
{
    std::string catname(category);
    journal::debug_t dbg(catname);
    int active = dbg.state();
    return active;
}

// version
// $Id: debuginfo.cc,v 1.1 2005/06/04 01:07:26 cummings Exp $

// End of file

