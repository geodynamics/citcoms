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


#include <Python.h>


// forward declarations
namespace journal {
    class Diagnostic;
}


class journal::Diagnostic {

// types
public:
    typedef std::string string_t;
    typedef std::stringstream buffer_t;

// interface
public:
    void state(bool flag) {
        PyObject *ret = PyObject_CallMethod(_self, (char *)(flag ? "activate" : "deactivate"), 0);
        if (ret) { Py_DECREF(ret); } else { _error(); }
    }
    bool state() const {
        PyObject *state = PyObject_GetAttrString(_self, (char *)"state"); if (!state) { _error(); }
        bool ret = false;
        switch (PyObject_IsTrue(state)) {
        case 1: ret = true; break;
        case -1: _error(); break;
        }
        Py_DECREF(state);
        return ret;
    }
    
    void activate() { state(true); }
    void deactivate() { state(false); }

    string_t facility() const { return _facility; }

    // entry manipulation
    void record() {
        newline();
        PyObject *ret = PyObject_CallMethod(_self, (char *)"record", 0);
        if (ret) { Py_DECREF(ret); } else { _error(); }
    }
    void newline() {
        string_t message = str();
        PyObject *ret = PyObject_CallMethod(_self, (char *)"line", (char *)"s#", message.c_str(), message.size());
        if (ret) { Py_DECREF(ret); } else { _error(); }
        _buffer.str(string_t());
    }
    void attribute(string_t key, string_t value) {
        PyObject *ret = PyObject_CallMethod(_self, (char *)"attribute", (char *)"s#s#",
                                            key.c_str(), key.size(),
                                            value.c_str(), value.size());
        if (ret) { Py_DECREF(ret); } else { _error(); }
    }
    void attribute(string_t key, long value) {
        PyObject *ret = PyObject_CallMethod(_self, (char *)"attribute", (char *)"s#l", key.c_str(), key.size(), value);
        if (ret) { Py_DECREF(ret); } else { _error(); }
    }

    // access to the buffered data
    string_t str() const { return _buffer.str(); }

    // builtin data type injection
    template <typename item_t> 
    Diagnostic & inject(item_t item) {
        _buffer << item;
        return *this;
    }

private:
    static PyObject *_journal() {
        static PyObject *journal;
        if (!journal) {
            journal = PyImport_ImportModule((char *)"journal");
            if (!journal) {
                PyErr_Print();
                Py_FatalError("could not import journal module");
            }
        }
        return journal;
    }
    static PyObject *_getDiagnostic(string_t facility, string_t severity) {
        PyObject *factory = PyObject_GetAttrString(_journal(), (char *)severity.c_str()); if (!factory) { _error(); }
        PyObject *diag = PyObject_CallFunction(factory, (char *)"s#", facility.c_str(), facility.size()); if (!diag) { _error(); }
        Py_DECREF(factory);
        return diag;
    }
    static void _error() {
        PyErr_Print();
        Py_Exit(1);
    }
    
// meta-methods
public:
    ~Diagnostic() { Py_DECREF(_self); }
    Diagnostic(string_t facility, string_t severity):
        _facility(facility), _severity(severity),
        _buffer(),
        _self(_getDiagnostic(facility, severity)) {
        if (PyErr_Occurred()) {
            _error();
        }
    }

// disable these
private:
    Diagnostic(const Diagnostic &);
    const Diagnostic & operator=(const Diagnostic &);

// data
private:
    const string_t _facility;
    const string_t _severity;

    buffer_t _buffer;

    PyObject *_self;
};


// the injection operator
template <typename item_t>
inline journal::Diagnostic & operator<< (journal::Diagnostic & diagnostic, item_t item) {
    return diagnostic.inject(item);
}


#endif

// version
// $Id: Diagnostic.h,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file 
