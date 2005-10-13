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


// meta-methods
Mesher::Mesher() :
    GLOBAL_MESH_MANAGER(),
    _nNodes(0),
    _nPolygons(0),
    _currentPolygon(0),
    _mesh(0)
{}

Mesher::~Mesher() 
{}

// interface
void Mesher::mesh(mesh_t * m)
{
    _mesh = m;
    return;
}

void Mesher::announce_counts(int npoly, int nnode, int npolynode)
{
    _nNodes = nnode;
    _nPolygons = npoly;

    _mesh->vertexCount(nnode);
    _mesh->simplexCount(npoly);

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

    // save the coordinates
    _mesh->vertex(node, 0, pos.x());
    _mesh->vertex(node, 1, pos.y());
    _mesh->vertex(node, 2, pos.z());

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

    // save the coordinates
    _mesh->vertex(node, 0, pos.x());
    _mesh->vertex(node, 1, pos.y());
    _mesh->vertex(node, 2, pos.z());

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

    // save the coordinates
    _mesh->vertex(node, 0, pos.x());
    _mesh->vertex(node, 1, pos.y());
    _mesh->vertex(node, 2, pos.z());

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

    _mesh->simplex(polygon, node, (long)id);

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


// version
// $Id: Mesher.cc,v 1.1.1.1 2005/03/08 16:13:32 aivazis Exp $

// End of file
