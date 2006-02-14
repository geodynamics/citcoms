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

#include "imports"
#include <vector>

#include "Mesher.h"


Mesher::Mesher() :
    GLOBAL_MESH_MANAGER(),
    _nNodes(0),
    _nPolygons(0),
    _nPolynodes(0),

    _nodes(0),
    _triangles(0),

    _currentPolygon(0)
{}

Mesher::~Mesher() 
{
    // remove leaks
    delete [] _nodes;
    delete [] _triangles;
}

void Mesher::announce_counts(int npoly, int nnode, int npolynode)
{
    _nNodes = nnode;
    _nPolygons = npoly;
    _nPolynodes = npolynode;

    _nodes = new position[_nNodes];
    _triangles = new long[3 * _nPolygons];

    return;
}

void * Mesher::announce_global_node(int node, VERTEX * v, const position & pos)
{
    if (node >= _nNodes) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "unexpected node id " << node << ", max = " << _nNodes - 1
            << journal::endl;
    }

    _nodes[node] = pos;

    return (void*)node;
}

void * Mesher::announce_global_node(int node, EDGE * e, const position & pos, double)
{
    if (node >= _nNodes) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "unexpected node id " << node << ", max = " << _nNodes - 1
            << journal::endl;
    }

    _nodes[node] = pos;

    return (void*)node;
}

void * Mesher::announce_global_node(int node, FACE * f, const position & pos, const par_pos &)
{
    if (node >= _nNodes) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "unexpected node id " << node << ", max = " << _nNodes - 1
            << journal::endl;
    }

    _nodes[node] = pos;

    return (void*)node;
}

void Mesher::start_indexed_polygon(int polygon, int nodes, int)
{
    _currentPolygon = polygon;

    if (polygon >= _nPolygons) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "unexpected polygon id " << polygon << ", max = " << _nPolygons - 1
            << journal::endl;
    }

    if (nodes != 3) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "a " << nodes << "-gon was announced; only triangles are supported"
            << journal::endl;
    }

    return;
}

void Mesher::announce_indexed_polynode(int polygon, int node, void * id)
{
    if (polygon >= _nPolygons) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "unexpected polygon id " << polygon << ", current = " << _currentPolygon
            << journal::endl;
    }

    _triangles[3*polygon + node] = (long)id;

    return;
}

void Mesher::end_indexed_polygon(int polygon)
{

    if (polygon >= _nPolygons) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "unexpected polygon id " << polygon << ", current = " << _currentPolygon
            << journal::endl;
    }

    return;
}


void Mesher::pack(char * buffer, size_t size) const
{
    size_t required = 
        4*sizeof(int) + 3*(_nNodes*sizeof(double) + _nPolygons*sizeof(int))
        + _nNodes*sizeof(double);

    if (size < required) {
        journal::firewall_t check("acis.mesher");
        check 
            << journal::at(__HERE__)
            << "insufficient buffer size: given " << size << ", required " << required
            << journal::endl;
    }

    size_t sz;
    char * cursor = buffer;

    // pack the endian hint
    int endian = 1;
    sz = sizeof(int);
    memcpy(cursor, &endian, sz);
    cursor += sz;

    // pack the number of vertices
    memcpy(cursor, &_nNodes, sz);
    cursor += sz;

    // pack the number of triangles
    memcpy(cursor, &_nPolygons, sz);
    cursor += sz;

    // pack the number of fields
    int fields = 1;
    memcpy(cursor, &fields, sz);
    cursor += sz;

    // pack the vertices
    sz = sizeof(double);
    for (int node = 0; node < _nNodes; ++node) {
        const position & pos = _nodes[node];
        double x = pos.x();
        double y = pos.y();
        double z = pos.z();

        memcpy(cursor, &x, sz);
        cursor += sz;
        memcpy(cursor, &y, sz);
        cursor += sz;
        memcpy(cursor, &z, sz);
        cursor += sz;
    }

    // pack the connectivity
    sz = sizeof(int);
    for (int node = 0; node < 3*_nPolygons; ++node) {
        int v = _triangles[node];
        memcpy(cursor, &v, sz);
        cursor += sz;
    }

    sz = sizeof(double);
    double value = 0;
    for (int node = 0; node < _nNodes; ++node) {
        memcpy(cursor, &value, sz);
        cursor += sz;
    }

    return;
}

void Mesher::dump() const
{
    journal::info_t info("acis.mesher.debug");

    info
        << journal::at(__HERE__)
        << "mesh: " << _nNodes << " vertices, " << _nPolygons << " triangles"
        << journal::newline;

    for (int node = 0; node < _nNodes; ++node) {
        const position & pos = _nodes[node];
        info
            << "node " << node << ": {" << pos.x() << ", " << pos.y() << ", " << pos.z() << "}" 
            << journal::newline;
    }

    for (int polygon = 0; polygon < _nPolygons; ++polygon) {
        long v0 = _triangles[3*polygon + 0];
        long v1 = _triangles[3*polygon + 1];
        long v2 = _triangles[3*polygon + 2];

        info
            << "triangle " << polygon << ": (" << v0 << ", " << v1 << ", " << v2 << ")"
            << journal::newline;
    }

    info << journal::endl;

    return;
}

// version
// $Id: Mesher-old.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
