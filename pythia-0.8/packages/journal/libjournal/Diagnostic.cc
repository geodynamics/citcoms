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

#include <portinfo>

#include <string>
#include <sstream>
#include <map>
#include <vector>

#include "Diagnostic.h"
#include "Facility.h"
#include "Entry.h"
#include "Device.h"
#include "Journal.h"

using namespace journal;

// interface
Diagnostic::journal_t & Diagnostic::journal() {
    // statics
    static Diagnostic::journal_t * _journal = new Journal();

    return *_journal;
}


bool Diagnostic::state() const {
    return _state.state();
}

void Diagnostic::state(bool flag) {
    _state.state(flag);
    return;
}

void Diagnostic::record() {
    if (!_state.state()) {
        return;
    }

    _newline();

    // record the journal entry
    journal().record(*_entry);

    // reset the entry
    delete _entry;
    _entry = new Entry;
    (*_entry)["severity"] = _severity;
    (*_entry)["facility"] = _facility;

    return;
}

void Diagnostic::newline() {
    if (!_state.state()) {
        return;
    }

    _newline();

    return;
}

void Diagnostic::attribute(string_t key, string_t value) {
    (*_entry)[key] = value;
    return;
}


// meta-methods
Diagnostic::~Diagnostic()
{
    delete _entry;
}

Diagnostic::Diagnostic(string_t facility, string_t severity, state_t & state):
    _facility(facility), _severity(severity),
    _state(state),
    _buffer(),
    _entry(new entry_t)
{
    (*_entry)["facility"] = facility;
    (*_entry)["severity"] = severity;
}


// implementation
void Diagnostic::_newline() {
    _entry->newline(str());
    _buffer.str(string_t());
    return;
}


// version
// $Id: Diagnostic.cc,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
