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

#include <portinfo>
#include <string>
#include <ostream>

#include "Device.h"
#include "StreamDevice.h"
#include "Console.h"

#include <iostream>

using namespace journal;

// meta-methods
journal::Console::Console() :
    StreamDevice(std::cout)
{}

Console::~Console() {}

// version
// $Id: Console.cc,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file
