/* -*- C++ -*-
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *                             Michael A.G. Aivazis
 *                      California Institute of Technology
 *                      (C) 1998-2005  All Rights Reserved
 *
 * <LicenseText>
 *
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 */


#if !defined(journal_firewall_h)
#define journal_firewall_h


/* get definition of __HERE__ macros */
#include "macros.h"


#ifdef __cplusplus
extern "C"
#endif
void firewall_hit(__HERE_DECL__, const char * fmt, ...);

#ifdef __cplusplus
extern "C"
#endif
void firewall_affirm(int condition, __HERE_DECL__, const char * fmt, ...);


#ifdef __cplusplus
#include <string>
#include <sstream>

#include "Diagnostic.h"
#include "SeverityFirewall.h"

#include "manipulators.h"

/* forward declarations */
namespace journal {

    typedef SeverityFirewall firewall_t;
}
#endif /* __cplusplus */

#endif /* journal_firewall_h */

/* version */
/* $Id: firewall.h,v 1.2 2005/06/04 01:07:26 cummings Exp $ */

/* End of file */
