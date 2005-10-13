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

#if !defined(journal_DefaultRenderer_h)
#define journal_DefaultRenderer_h

// forward declarations
namespace journal {
    class Entry;
    class Renderer;
    class DefaultRenderer;
}

// 
class journal::DefaultRenderer : public journal::Renderer
{
// meta-methods
public:
    virtual ~DefaultRenderer();
    inline DefaultRenderer();

// implementation
protected:
    virtual string_t header(const Entry &);
    virtual string_t body(const Entry &);
    virtual string_t footer(const Entry &);

// disable these
private:
    DefaultRenderer(const DefaultRenderer &);
    const DefaultRenderer & operator=(const DefaultRenderer &);
};

#endif

// get the inline definitions
#define journal_DefaultRenderer_icc
#include "DefaultRenderer.icc"
#undef journal_DefaultRenderer_icc

// version
// $Id: DefaultRenderer.h,v 1.1.1.1 2005/03/08 16:13:56 aivazis Exp $

// End of file
