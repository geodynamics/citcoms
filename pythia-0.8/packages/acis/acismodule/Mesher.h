// -*- C++ -*-
//
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// 
//                               Michael A.G. Aivazis
//                        California Institute of Technology
//                        (C) 1998-2005 All Rights Reserved
// 
//  <LicenseText>
// 
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(pyacis_Mesher_h)
#define pyacis_Mesher_h

// ACIS includes
#include <faceter/meshmgr/gmeshmg.hxx>
#include "pyre/geometry/CanonicalMesh.h"

class position;

class Mesher : public GLOBAL_MESH_MANAGER
{
// types
public:
    typedef long * triangles_t;
    typedef position * nodes_t;

    typedef pyre::geometry::CanonicalMesh<double> mesh_t;

// interface
public:
    void mesh(mesh_t *);

// interface -- called during the meshing process
public:
    virtual void announce_counts(int polygons, int nodes, int polynodes);

    virtual void start_indexed_polygon(int polygon, int nodes, int);
    virtual void announce_indexed_polynode(int polygon, int node, void * id);
    virtual void end_indexed_polygon(int polygon);

    virtual void * announce_global_node(int node, VERTEX * v, const position &pos);
    virtual void * announce_global_node(int inode, EDGE * e, const position & pos, double);
    virtual void * announce_global_node(int node, FACE * f, const position & pos, const par_pos &);

// meta-methods
public:
    Mesher();
    virtual ~Mesher();

// data
private:

    // bookkeeping
    int _nNodes;
    int _nPolygons;
    int _currentPolygon;

    mesh_t * _mesh; // of 3-dimensional triangles

// hide these
private:
    Mesher(const Mesher &);
    const Mesher & operator=(const Mesher &);
};

#endif

// version
// $Id: Mesher.h,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
