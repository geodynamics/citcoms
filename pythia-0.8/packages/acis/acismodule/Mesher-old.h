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

class position;

class Mesher : public GLOBAL_MESH_MANAGER
{
// types
public:
    typedef long * triangles_t;
    typedef position * nodes_t;
    // typedef std::vector<long *> triangles_t;
    // typedef std::vector<position *> nodes_t;

public:

    Mesher();
    virtual ~Mesher();

    int nodes() const { return _nNodes; }
    int triangles() const { return _nPolygons; }

    void pack(char * buffer, size_t size) const;

    void start_indexed_polygon(int polygon, int nodes, int);
    void announce_indexed_polynode(int polygon, int node, void * id);
    void end_indexed_polygon(int polygon);

    virtual void announce_counts(int polygons, int nodes, int polynodes);
    void * announce_global_node(int node, VERTEX * v, const position &pos);
    void * announce_global_node(int inode, EDGE * e, const position & pos, double);
    void * announce_global_node(int node, FACE * f, const position & pos, const par_pos &);

    void dump() const;

private:

    int _nNodes;
    int _nPolygons;
    int _nPolynodes;

    nodes_t _nodes;
    triangles_t _triangles;

    int _currentPolygon;

private:

    Mesher(const Mesher &);
    const Mesher & operator=(const Mesher &);
};

#endif

// version
// $Id: Mesher-old.h,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
