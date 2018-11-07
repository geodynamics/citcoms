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

#if !defined(journal_manipulators_h)
#define journal_manipulators_h


namespace journal {

    // declarations of the builtin manipulators
    class set_t;
    class loc2_t;
    class loc3_t;
}


class journal::set_t {
// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, const char *, const char *);

// meta-methods
public:
    set_t(factory_t f, const char * key, const char * value) :
        _f(f), _key(key), _value(value) {}

// data
public:
    factory_t _f;
    const char * _key;
    const char * _value;
};


class journal::loc2_t {

// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, const char *, long);

// meta-methods
public:
    loc2_t(factory_t f, const char * file, long line) :
        _f(f), _file(file), _line(line) {}

// data
public:
    factory_t _f;
    const char * _file;
    long _line;
};


class journal::loc3_t {
// types
public:
    typedef Diagnostic & (*factory_t)(Diagnostic &, const char *, long, const char *);

// meta-methods
public:
    loc3_t(factory_t f, const char * file, long line, const char * function) :
        _f(f), _file(file), _line(line), _function(function) {}

// data
public:
    factory_t _f;
    const char * _file;
    long _line;
    const char * _function;
};


// the injection operators: leave these in the global namespace

inline journal::Diagnostic & operator<< (journal::Diagnostic & s, journal::set_t m)
{
    return (*m._f)(s, m._key, m._value);
}

inline journal::Diagnostic & operator<< (journal::Diagnostic & s, journal::loc2_t m)
{
    return (*m._f)(s, m._file, m._line);
}

inline journal::Diagnostic & operator<< (journal::Diagnostic & s, journal::loc3_t m)
{
    return (*m._f)(s, m._file, m._line, m._function);
}


// forward declarations
namespace journal {
    class Diagnostic;

    // end of entry
    inline Diagnostic & endl(Diagnostic & diag) {
        diag.record();
        return diag;
    }
    
    // add a newline
    inline Diagnostic & newline(Diagnostic & diag) {
        diag.newline();
        return diag;
    }

    // set metadata key to value
    inline Diagnostic & __diagmanip_set(Diagnostic & s, const char * key, const char * value) {
        s.attribute(key, value);
        return s;
    }

    inline set_t set(const char * key, const char * value) {
        return set_t(__diagmanip_set, key, value);
    }
    
    // location information
    inline Diagnostic & __diagmanip_loc(Diagnostic & s, const char * filename, long line) {
        s.attribute("filename", filename);
        s.attribute("line", line);
        return s;
    }

    inline loc2_t at(const char * file, long line) {
        return loc2_t(__diagmanip_loc, file, line);
    }

    inline Diagnostic & __diagmanip_loc(
        Diagnostic & s, const char * filename, long line, const char * function) 
    {
        s.attribute("filename", filename);
        s.attribute("function", function);
        s.attribute("line", line);
        return s;
    }

    inline loc3_t at(const char * file, long line, const char * function) {
        return loc3_t(__diagmanip_loc, file, line, function);
    }

}

inline journal::Diagnostic & 
operator<< (journal::Diagnostic & s, journal::Diagnostic & (m)(journal::Diagnostic &))
{
    return (*m)(s);
}


#endif

// version
// $Id: manipulators.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file 
