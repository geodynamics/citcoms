// -*- C++ -*-
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
//  <LicenseText>
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <iostream>
#include "global_defs.h"
#include "Boundary.h"
using namespace std;
// #include "ExchangerClass.h"

//using std::auto_ptr;

Boundary::Boundary(const int n) : size(n){
    std::cout << "in Boundary::Boundary  size = " << size << std::endl;

    connectivity = new int[size];
    for(int i=0; i<dim; i++)
	X[i] = new double[size];

    bid2proc = new int[size];
    bid2gid = new int[size];
    bid2crseelem[0] = new int[size];
    bid2crseelem[1] = new int[size];
    shape = new double[size*8];    
  
    // use auto_ptr for exception-proof
    //Boundary(n, auto_ptr<int>(new int[size]),
    //     auto_ptr<double>(new double[size]),
    //     auto_ptr<double>(new double[size]),
    //     auto_ptr<double>(new double[size]));
}


// // because initialization of X[dim] involves a loop,
// // here we assume dim=3 and use a private constructor

// Boundary::Boundary(const int n,
// 		   auto_ptr<int> c,
// 		   auto_ptr<double> x0,
// 		   auto_ptr<double> x1,
// 		   auto_ptr<double> x2)
//     :    size(n), connectivity_(c),
//         X_[0](x0), X_[1](x1), X_[2](x2)
// {
//     std::cout << "in Boundary::Boundary  size = " << size << std::endl;
//     assert(dim == 3);

//     // setup traditional pointer for convenience
//     connectivity = connectivity_.get();
//     for(int i=0; i<dim; i++)
// 	X[i] = X_[i].get();

// }



Boundary::~Boundary() {
    std::cout << "in Boundary::~Boundary" << std::endl;

    delete [] connectivity;
    for(int i=0; i<dim; i++)
	delete [] X[i];

    delete [] bid2proc;
    delete [] bid2gid;

    // memory allocated to Data structures
//     delete outgoing->T;
//     delete incoming->T;
//     delete outgoing->v[0];
//     delete outgoing->v[1];
//     delete outgoing->v[2];
//     delete incoming->v[0];
//     delete incoming->v[1];
//     delete incoming->v[2];
//     delete loutgoing;
//     delete lincoming;
};


void Boundary::init(const All_variables *E) {
    int nodes,node1,node2,nodest;
    int *nid;
    double theta_max,theta_min,fi_max,fi_min,ro,ri;

    nodest = E->lmesh.nox * E->lmesh.noy * E->lmesh.noz;
    nid = new int[nodest];
    for(int i=0;i<nodest;i++)nid[i]=0;

    nodes=0;
        // test
    for(int j=0; j<size; j++)
        connectivity[j] = j;

    theta_max=E->control.theta_max;
    theta_min=E->control.theta_min;
    fi_max=E->control.fi_max;
    fi_min=E->control.fi_min;
    ro=E->sphere.ro;
    ri=E->sphere.ri;

    // test 
    std::cout << "Fine Grid Bounds" << std::endl;
    std::cout << "theta= " << theta_min<< "   " << theta_max << std::endl;
    std::cout << "fi   = " << fi_min << "   " << fi_max << std::endl;
    std::cout << "r    = " << ri << "   " << ro  << std::endl;


    //  for two YOZ planes

    if (E->parallel.me_loc[1]==0 || E->parallel.me_loc[1]==E->parallel.nprocx-1) 
      for (int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int j=1;j<=E->lmesh.noy;j++)
	  for(int i=1;i<=E->lmesh.noz;i++)  {
	    node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
	    node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;
	    
	    if ((E->parallel.me_loc[1]==0) && (!nid[node1-1]))  {
	      for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node1];
	      nid[node1-1]++;
	      nodes++;
	    }
	    if ((E->parallel.me_loc[1]==E->parallel.nprocx-1) && (!nid[node2-1])) {
	      for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	      nid[node2-1]++;
	      nodes++;
	    }
	  }
    
    //  for two XOZ planes
    
    if (E->parallel.me_loc[2]==0 || E->parallel.me_loc[2]==E->parallel.nprocy-1)
      for (int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int j=1;j<=E->lmesh.nox;j++)
	  for(int i=1;i<=E->lmesh.noz;i++)  {
	    node1 = i + (j-1)*E->lmesh.noz;
	    node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
	    if ((E->parallel.me_loc[2]==0) && (!nid[node1-1]))  {
	      for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node1];
	      nid[node1-1]++;
	      nodes++;
	    }
	    if((E->parallel.me_loc[2]==E->parallel.nprocy-1)&& (!nid[node2-1]))  {
	      for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	      nid[node2-1]++;
	      nodes++;
	    }
	  }
    //  for two XOY planes
    if (E->parallel.me_loc[3]==0 || E->parallel.me_loc[3]==E->parallel.nprocz-1)
      for (int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int j=1;j<=E->lmesh.noy;j++)
	  for(int i=1;i<=E->lmesh.nox;i++)  {
	    node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
	    node2 = node1 + E->lmesh.noz-1;
	    
	    if ((E->parallel.me_loc[3]==0 ) && (!nid[node1-1])) {
	      for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node1];
	      nid[node1-1]++;
	      nodes++;
	    }
	    if ((E->parallel.me_loc[3]==E->parallel.nprocz-1) &&(!nid[node2-1])) {
	      for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	      nid[node2-1]++;
	      nodes++;
	    }
	  }
    if(nodes != size) std::cout << " nodes != size ";
    
    delete nid;
    return;
}


void Boundary::getBid2crseelem(const All_variables *E) {
    
   int ind,n,n1,n2;    
   double xt[3],xc[24],dett,det[4],x1[3],x2[3],x3[3],x4[3],xi[3],norm;
   int nsub[]={0, 2, 3, 7, 0, 1, 2, 5, 4, 7, 5, 0, 5, 7, 6, 2, 5, 7, 2, 0};
//   int nxmax,nxmin,nymax,nymin,nzmax,nzmin;
  
//   for(int j=1; j < 2; j++) {
//  for(int j=0; j < size; j++) {
//    bid2crseelem[j]=0;
//    ind=0;
//    for(int i=1; i <= E->lmesh.nel; i++)
//      {
//	nxmax = E->IEN[E->mesh.levmax][1][i].node[2];
//	nxmin = E->IEN[E->mesh.levmax][1][i].node[1];
//	nymax = E->IEN[E->mesh.levmax][1][i].node[4];
//	nymin = E->IEN[E->mesh.levmax][1][i].node[1];
//	nzmax = E->IEN[E->mesh.levmax][1][i].node[5];
//	nzmin = E->IEN[E->mesh.levmax][1][i].node[1];
// Test
// 	std::cout << "in Boundary::init " << nxmax << " "  << nxmin << " " 
// 		  << nymax << " "  << nymin << " " 
// 		  << nzmax << " "  << nzmin << " | " 
// 		  << X[0][j] << " "  
// 		  << X[1][j] << " "  
// 		  << X[2][j] << " | "  
// 		  << E->X[E->mesh.levmax][1][1][nxmax] << " "  << E->X[E->mesh.levmax][1][1][nxmin] << " " 
// 		  << E->X[E->mesh.levmax][1][2][nymax] << " "  << E->X[E->mesh.levmax][1][2][nymin] << " " 
// 		  << E->X[E->mesh.levmax][1][3][nzmax] << " "  << E->X[E->mesh.levmax][1][3][nzmin] << " " 
// 		  << std::endl;

//	if(  (X[0][j] >= E->X[E->mesh.levmax][1][1][nxmin]) 
//	     &&(X[0][j] <= E->X[E->mesh.levmax][1][1][nxmax])
//	     &&(X[1][j] >= E->X[E->mesh.levmax][1][2][nymin])
//	     &&(X[1][j] <= E->X[E->mesh.levmax][1][2][nymax])
//	     &&(X[2][j] >= E->X[E->mesh.levmax][1][3][nzmin])
//	     &&(X[2][j] <= E->X[E->mesh.levmax][1][3][nzmax]) )
//	  {
//	    bid2crseelem[0][j]=i;
//	    ind=1;
//	  }
//	if(ind) {
//	  std::cout << " done right bid = " << j << " "
//		    << " bid2crseelem[j] = " << bid2crseelem[0][j]
//		    << std::endl;
//	  break;
//	}
//      }
//    if(!ind)
//      std::cout << "  wrong bid = " << j << std::endl;
//  }
  for(int i=0; i< size; i++)
    {
        for(int j=0; j< dim; j++)xt[j]=X[j][i];
// loop over 5 sub tets in a brick element
        ind = 0;
         
        for(int mm=1;mm<=E->sphere.caps_per_proc;mm++)
            for(n=0;n<E->lmesh.nel;n++)
            {
                for(int j=0; j < 8; j++)
                    for(int k=0; k < dim; k++)
                    {
                        xc[j*dim+k]=E->X[E->mesh.levmax][mm][k+1][E->IEN[E->mesh.levmax][mm][n+1].node[j+1]];
                    }
                  
                for(int k=0; k < 5; k++)
                {
                    for(int m=0; m < dim; m++)
                    {
                        x1[m]=xc[nsub[k*4]*dim+m];
                        x2[m]=xc[nsub[k*4+1]*dim+m];
                        x3[m]=xc[nsub[k*4+2]*dim+m];
                        x4[m]=xc[nsub[k*4+3]*dim+m];
                    }
                    dett=Tetrahedronvolume(x1,x2,x3,x4);
                    det[0]=Tetrahedronvolume(x2,x4,x3,xt);
                    det[1]=Tetrahedronvolume(x3,x4,x1,xt);
                    det[2]=Tetrahedronvolume(x1,x4,x2,xt);
                    det[3]=Tetrahedronvolume(x1,x2,x3,xt);
                    if(dett < 0) std::cout << " Determinent evaluation is wrong" << std::endl;
                    if(det[0] < 0.0 || det[1] <0.0 || det[2] < 0.0 || det[3] < 0.0) continue;                    
                    ind=1;
                    bid2crseelem[0][i]=n+1;
                    bid2crseelem[1][i]=mm;
                    
                    shape[i*8+nsub[k*4]]=det[0]/dett;
                    shape[i*8+nsub[k*4+1]]=det[1]/dett;
                    shape[i*8+nsub[k*4+2]]=det[2]/dett;
                    shape[i*8+nsub[k*4+3]]=det[3]/dett;
                                       
                    break;
                    
                }
                if(ind)
                {
                    xi[0]=xi[1]=xi[2]=0.0;
                    for(int j=0; j < 8; j++)
                        for(int k=0; k < dim; k++)
                        {
                  
                            xi[k]+=xc[j*dim+k]*shape[i*8+j];
                        }
                }
                
                if(ind) break;          
            }
    }
// test
//  std::cout << "in Boundary::bid2crseelem for bid2crseelem " << std::endl;    
    for(int i=0; i< size; i++)
    {
        n1=bid2crseelem[0][i];
        n2=bid2crseelem[1][i];
        for(int j=0; j< dim; j++)xt[j]=X[j][i];
        for(int j=0; j < 8; j++)
        {
            
            for(int k=0; k < dim; k++)
            {                
                xc[j*dim+k]=E->X[E->mesh.levmax][n2][k+1][E->IEN[E->mesh.levmax][n2][n1].node[j+1]];
            }
//            std::cout <<" " <<xc[j*dim] << " " << xc[j*dim+1] << " " << xc[j*dim+2] <<" "<< shape[i*8+j] << std::endl;
        }        
        for(int k=0; k<dim; k++)xi[k]=0.0;
        for(int k=0; k<dim; k++)
            for(int j=0; j < 8; j++)
            {
                xi[k]+=xc[j*dim+k]*shape[i*8+j];                
            }
//        std::cout << " "<< xt[0] <<" "<< xi[0] <<" "<< xt[1] << " "<< xi[1] << " " << xt[2] << " " << xi[2] << std::endl;
        norm = 0.0;
        for(int k=0; k < dim; k++) norm+=(xt[k]-xi[k])*(xt[k]-xi[k]);
        if(norm > 1.e-10)
        {            
            std::cout << "\n in Boundary::mapCoarseGrid for bid2crseelem interpolation functions are wrong " << norm << std::endl;
        }
        
    }
//    std::cout << "end of  Boundary::bid2crseelem for bid2crseelem " << std::endl; 
    return;
}



// void Boundary::mapFineGrid(const All_variables *E, int localLeader) {

//     for(int i=0; i<size; i++)
//         bid2proc[i]=localLeader;

//     int n=0;
//     for(int m=1;m<=E->sphere.caps_per_proc;m++)
// 	for(int k=1;k<=E->lmesh.noy;k++)
// 	    for(int j=1;j<=E->lmesh.nox;j++)
// 		for(int i=1;i<=E->lmesh.noz;i++)  {
// 		    int node = i + (j-1)*E->lmesh.noz
//                         + (k-1)*E->lmesh.noz*E->lmesh.nox;

// 		    if((k==1)||(k==E->lmesh.noy)||(j==1)||(j==E->lmesh.nox)||(i==1)||(i==E->lmesh.noz))
// 		    {
//                         bid2gid[n]=node;
//                         n++;
// 		    }
// 		}
//     if(n != size) std::cout << " nodes != size ";
//     printBid2gid();
// }

void Boundary::mapFineGrid(const All_variables *E, int localLeader) {
    int nodes,node1,node2,nodest;
    int *nid;
    // xc is the array of the coarse grid coordinates
    //  double xt[3],xc[24],xl[12],dett,det[4];
//      int nsub[]={1, 3, 4, 8, 1, 2, 3, 6, 5, 8, 6, 1, 6, 8, 7, 3, 6, 8, 3, 1};  
  
  for(int i=0; i<size; i++)
    bid2proc[i]=localLeader;
  
  
  nodest = E->lmesh.nox * E->lmesh.noy * E->lmesh.noz;
  nid = new int[nodest];
  for(int i=0;i<nodest;i++)nid[i]=0;

  nodes=0;
  
  //  for two YOZ planes
  
  if (E->parallel.me_loc[1]==0 || E->parallel.me_loc[1]==E->parallel.nprocx-1) 
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
      for(int j=1;j<=E->lmesh.noy;j++)
	for(int i=1;i<=E->lmesh.noz;i++)  {
	  node1 = i + (j-1)*E->lmesh.noz*E->lmesh.nox;
	  node2 = node1 + (E->lmesh.nox-1)*E->lmesh.noz;
	  
	  if ((E->parallel.me_loc[1]==0) && (!nid[node1-1]))  {
	    bid2gid[nodes] = node1;
	    nodes++;
	    nid[node1-1]++;
	  }
	  if ((E->parallel.me_loc[1]==E->parallel.nprocx-1) && (!nid[node2-1])) {
	    bid2gid[nodes] = node2;
	    nodes++;
	    nid[node2-1]++;
	  }
	}
  
  //  for two XOZ planes
  
  if (E->parallel.me_loc[2]==0 || E->parallel.me_loc[2]==E->parallel.nprocy-1)
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
      for(int j=1;j<=E->lmesh.nox;j++)
	for(int i=1;i<=E->lmesh.noz;i++)  {
	  node1 = i + (j-1)*E->lmesh.noz;
	  node2 = node1 + (E->lmesh.noy-1)*E->lmesh.noz*E->lmesh.nox;
	  if ((E->parallel.me_loc[2]==0) && (!nid[node1-1]))  {
	    bid2gid[nodes] = node1;
	    nodes++;
	    nid[node1-1]++;
	  }
	  if((E->parallel.me_loc[2]==E->parallel.nprocy-1)&& (!nid[node2-1]))  {
	    for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][1][k+1][node2];
	    bid2gid[nodes] = node2;
	    nodes++;
	    nid[node2-1]++;
	  }
	}
  //  for two XOY planes
  if (E->parallel.me_loc[3]==0 || E->parallel.me_loc[3]==E->parallel.nprocz-1)
    for (int m=1;m<=E->sphere.caps_per_proc;m++)
      for(int j=1;j<=E->lmesh.noy;j++)
	for(int i=1;i<=E->lmesh.nox;i++)  {
	  node1 = 1 + (i-1)*E->lmesh.noz+(j-1)*E->lmesh.nox*E->lmesh.noz;
	  node2 = node1 + E->lmesh.noz-1;
	  
	  if ((E->parallel.me_loc[3]==0 ) && (!nid[node1-1])) {
	    for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][m][k+1][node1];
	    bid2gid[nodes] = node1;
	    nodes++;
	    nid[node1-1]++;
	  }
	  if ((E->parallel.me_loc[3]==E->parallel.nprocz-1) &&(!nid[node2-1])) {
	    for(int k=0;k<dim;k++)X[k][nodes]=E->X[E->mesh.levmax][m][k+1][node2];
	    bid2gid[nodes] = node2;
	    nodes++;
	    nid[node2-1]++;
	  }
	}
  if(nodes != size) cout << "in Boundary::mapFineGrid ==> nodes != size " << endl;

  delete nid;
  
  return;
}
double Boundary::Tetrahedronvolume(double  *x1, double *x2, double *x3, double *x4)
{
    double vol;
//    xx[0] = x2;  xx[1] = x3;  xx[2] = x4;
    vol = det3_sub(x2,x3,x4);
//    xx[0] = x1;  xx[1] = x3;  xx[2] = x4;
    vol -= det3_sub(x1,x3,x4);
//    xx[0] = x1;  xx[1] = x2;  xx[2] = x4;
    vol += det3_sub(x1,x2,x4);
//    xx[0] = x1;  xx[1] = x2;  xx[2] = x3;
    vol -= det3_sub(x1,x2,x3);
    vol /= 6.;
    return vol;       
}
double Boundary::det3_sub(double *x1, double *x2, double *x3)
{
    return (x1[0]*(x2[1]*x3[2]-x3[1]*x2[2])
            -x1[1]*(x2[0]*x3[2]-x3[0]*x2[2])
            +x1[2]*(x2[0]*x3[1]-x3[0]*x2[1]));
//   return (x[0][0]*(x[1][1]*x[2][2]-x[2][1]*x[1][2])
//              -x[0][1]*(x[1][0]*x[2][2]-x[2][0]*x[1][2])
//              +x[0][2]*(x[1][0]*x[2][1]-x[2][0]*x[1][1]));
}


void Boundary::mapCoarseGrid(const All_variables *E, int localLeader) {
//      int ind,n1,n2;
//     double xt[3],xc[24],dett,det[4],x1[3],x2[3],x3[3],x4[3],xi[3],norm;
//      int nsub[]={0, 2, 3, 7, 0, 1, 2, 5, 4, 7, 5, 0, 5, 7, 6, 2, 5, 7, 2, 0};
    
    for(int i=0; i<size; i++)
        bid2proc[i]=localLeader;

    int n=0;
    for(int m=1;m<=E->sphere.caps_per_proc;m++)
	for(int k=1;k<=E->lmesh.noy;k++)
	    for(int j=1;j<=E->lmesh.nox;j++)
		for(int i=1;i<=E->lmesh.noz;i++)  {
		    int node = i + (j-1)*E->lmesh.noz
                        + (k-1)*E->lmesh.noz*E->lmesh.nox;

		    if((k==1)||(k==E->lmesh.noy)||(j==1)||(j==E->lmesh.nox)||(i==1)||(i==E->lmesh.noz))
		    {
                        bid2gid[n]=node;
                        n++;
		    }
		}
    if(n != size) std::cout << " nodes != size ";
    printBid2gid();
 
    return;  
}


void Boundary::printConnectivity() const {
    int *c = connectivity;
    for(int j=0; j<size; j++)
	std::cout << "  C:  " << j << ":  " << c[j] << std::endl;
}


void Boundary::printX() const {
    for(int j=0; j<size; j++) {
	std::cout << "  X:  " << j << ":  ";
	    for(int i=0; i<dim; i++)
		std::cout << X[i][j] << " ";
	std::cout << std::endl;
    }
}


void Boundary::printBid2gid() const {
    int *c = bid2gid;
    for(int j=0; j<size; j++)
	std::cout << "  B:  " << j << ":  " << c[j] << std::endl;
}



// version
// $Id: Boundary.cc,v 1.16 2003/09/24 20:14:12 puru Exp $

// End of file
