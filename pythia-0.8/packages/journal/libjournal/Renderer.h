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

#if !defined(journal_Renderer_h)
#define journal_Renderer_h

// forward declarations
namespace journal {
    class Entry;
    class Renderer;
}

class journal::Renderer
{
//types
public:
    typedef std::string string_t;

// interface
public:
    virtual string_t render(const Entry &);

// meta-methods
public:
    virtual ~Renderer();
    inline Renderer();

// implementation
protected:
    virtual string_t header(const Entry &) = 0;
    virtual string_t body(const Entry &) = 0;
    virtual string_t footer(const Entry &) = 0;

// disable these
private:
    Renderer(const Renderer &);
    const Renderer & operator=(const Renderer &);
};

#endif

// get the inline definitions
#define journal_Renderer_icc
#include "Renderer.icc"
#undef journal_Renderer_icc

// version
// $Id: Renderer.h,v 1.1.1.1 2005/03/08 16:13:55 aivazis Exp $

// End of file
