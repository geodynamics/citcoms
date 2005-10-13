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

#if !defined(journal_Diagnostic_h)
#define journal_Diagnostic_h


// forward declarations
namespace journal {
    class Entry;
    class Facility;
    class Journal;
    class Diagnostic;
}

// injection operator
template <typename item_t>
inline journal::Diagnostic & operator<< (journal::Diagnostic &, item_t);


class journal::Diagnostic {

// types
public:
    typedef Entry entry_t;
    typedef Facility state_t;
    typedef Journal journal_t;

    typedef std::string string_t;
    typedef std::stringstream buffer_t;

// interface
public:
    void state(bool);
    bool state() const;

    inline void activate();
    inline void deactivate();

    inline string_t facility() const;

    // entry manipulation
    void record();
    void newline();
    void attribute(string_t, string_t);

    // access to the buffered data
    inline string_t str() const;

    // access to the journal singleton
    static journal_t & journal();

    // builtin data type injection
    template <typename item_t> 
    inline Diagnostic & inject(item_t datum);

// meta-methods
public:
    ~Diagnostic();
    Diagnostic(string_t, string_t, state_t &);

// implementation
private:
    void _newline();

// disable these
private:
    Diagnostic(const Diagnostic &);
    const Diagnostic & operator=(const Diagnostic &);

// data
private:
    const string_t _facility;
    const string_t _severity;

    state_t & _state;
    buffer_t _buffer;
    entry_t * _entry;
};

// get the inline definitions
#define journal_Diagnostic_icc
#include "Diagnostic.icc"
#undef journal_Diagnostic_icc

#endif

// version
// $Id: Diagnostic.h,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file 
