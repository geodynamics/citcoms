// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#include <portinfo>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include "journal/journal.h"
#include "BoundedMesh.h"
#include "Sink.h"


Sink::Sink(MPI_Comm c, int nsrc, const BoundedMesh& mesh) :
    comm(c),
    numSrcNodes(nsrc),
    beginSrcNodes(nsrc+1),
    numMeshNodes(mesh.size())
{
    checkCommSize(nsrc);

    MPI_Comm_rank(comm, &me);

    source.reserve(nsrc);
    for(int i=0; i<nsrc+1; i++)
	if(i != me)
	    source.push_back(i);

    sendMesh(mesh);

    recvMeshNode();
    initX(mesh);
}


// private functions

void Sink::checkCommSize(int nsrc) const
{
    int size;
    MPI_Comm_size(comm, &size);
    if(size != nsrc + 1) {
	journal::firewall_t firewall("Sink");
	firewall << journal::loc(__HERE__)
		 << "size of communicator != (numSrc + 1)" << journal::end;
	throw std::domain_error("size of communicator != (numSrc + 1)");
    }
}


void Sink::sendMesh(const BoundedMesh& mesh) const
{
    // assuming sources are listening to broadcast
    mesh.broadcast(comm, me);
}


void Sink::recvMeshNode()
{
    recvSourceSize();
    sumSourceSize();

    // fill meshNode_ with value "numMeshNodes", which is an invalid node #
    meshNode_.resize(beginSrcNodes[source.size()], numMeshNodes);

    recv(meshNode_);
    meshNode_.print("meshNode_recv");

    testMeshNode();
}


void Sink::recvSourceSize()
{
    for(size_t i=0; i<source.size(); i++)
 	numSrcNodes[i] = meshNode_.receiveSize(comm, source[i]);
}


void Sink::sumSourceSize()
{
    partial_sum(numSrcNodes.begin(), numSrcNodes.end(),
		++beginSrcNodes.begin());

    journal::debug_t debug("Exchanger");
    for(size_t i=0; i<source.size(); i++) {
	debug << journal::loc(__HERE__)
	      << " source = " << i << "  size = " << numSrcNodes[i]
	      << "  begin = " << beginSrcNodes[i] << journal::newline;
    }
    debug << " total nodes = " << beginSrcNodes[source.size()]
	  << journal::end;
}


void Sink::testMeshNode() const
{
    if(std::find(meshNode_.begin(), meshNode_.end(), numMeshNodes)
       != meshNode_.end()) {
	journal::firewall_t firewall("Sink");
	firewall << journal::loc(__HERE__)
		 << "some node in meshNode not mapped" << journal::end;
	throw std::domain_error("Sink");
    }
}


void Sink::initX(const BoundedMesh& mesh)
{
    X_.resize(meshNode_.size());

    for(int i=0; i<X_.size(); ++i) {
 	int n = meshNode_[0][i];
	for(int j=0; j<DIM; ++j)
	    X_[j][i] = mesh.X(j,n);
    }
}


// version
// $Id: Sink.cc,v 1.6 2004/03/11 22:42:17 tan2 Exp $

// End of file
