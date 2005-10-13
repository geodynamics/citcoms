// -*- C++ -*-
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Julian C. Cummings
//                       California Institute of Technology
//                        (C) 1998-2003 All Rights Reserved
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

#include <cstdarg>
#include <cstdio>
#include <string>

#include "firewall.h"


extern "C"
void firewall_hit(__HERE_DECL__, const char * fmt, ...)
{
    // create Firewall object and log message
    // use filename as name, since we have no category arg
    std::string name(filename);
    journal::firewall_t fwall(name);

    std::va_list args;
    char buffer[2048];
    va_start(args, fmt);
    std::vsprintf(buffer, fmt, args);
    va_end(args);

    fwall << journal::at(__HERE_ARGS__)
          << buffer << journal::endl;

    return;
}



extern "C"
void firewall_affirm(int condition, __HERE_DECL__, const char * fmt, ...)
{
    if (!condition) {
        // create Firewall object and log message
        // use filename as name, since we have no category arg
        std::string name(filename);
        journal::firewall_t fwall(name);

        std::va_list args;
        char buffer[2048];
        va_start(args, fmt);
        std::vsprintf(buffer, fmt, args);
        va_end(args);

        fwall << journal::at(__HERE_ARGS__)
              << buffer << journal::endl;
    }

    return;
}

// version
// $Id: firewall.cc,v 1.1 2005/06/04 01:07:26 cummings Exp $

// End of file
