/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 *<LicenseText>
 *
 * CitcomS by Louis Moresi, Shijie Zhong, Lijie Han, Eh Tan,
 * Clint Conrad, Michael Gurnis, and Eun-seo Choi.
 * Copyright (C) 1994-2005, California Institute of Technology.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 *</LicenseText>
 *
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "parsing.h"
#include "parallel_related.h"
#include "composition_related.h"

static void get_2dshape(struct All_variables *E,
                        int j, ElementID nelem,
                        CoordUV uv,
                        int iwedge, double * shape2d);
static void get_radial_shape(struct All_variables *E,
                             int j, ElementID nelem,
                             double rad, double *shaperad);
static CoordUV spherical_to_uv(struct All_variables *E,
                            SphericalCoord sc);
static int icheck_column_neighbors(struct All_variables *E,
                                   int j, ElementID nel,
                                   CartesianCoord cc,
                                   double rad);
static int icheck_all_columns(struct All_variables *E,
                              int j,
                              CartesianCoord cc,
                              double rad);
static int icheck_element(struct All_variables *E,
                          int j, ElementID nel,
                          CartesianCoord cc,
                          double rad);
static int icheck_shell(struct All_variables *E,
                        ElementID nel, double rad);
static int icheck_element_column(struct All_variables *E,
                                 int j, ElementID nel,
                                 CartesianCoord cc,
                                 double rad);
static int icheck_bounds(struct All_variables *E,
                         CartesianCoord test_point,
                         const CapBoundary bounds);
static double findradial(CartesianCoord vec,
                         BoundaryPoint bp);
static void fix_angle(double *angle);
static int iget_radial_element(struct All_variables *E,
                               int j, int iel,
                               double rad);
static int iget_regel(struct All_variables *E, int j,
                      double theta, double phi,
                      int *ntheta, int *nphi);
static void full_put_lost_tracers(struct All_variables *E,
                                  int isend[13][13], double *send[13][13]);
void pdebug(struct All_variables *E, int i);
int full_icheck_cap(struct All_variables *E, int icap,
                    CartesianCoord cc, double rad);


/******* FULL TRACER INPUT *********************/

void full_tracer_input(struct All_variables *E)
{
    int m = E->parallel.me;

    /* Regular grid parameters */
    /* (first fill uniform del[0] value) */
    /* (later, in make_regular_grid, will adjust and distribute to caps */

    E->trace.deltheta[0]=1.0;
    E->trace.delphi[0]=1.0;
    input_double("regular_grid_deltheta",&(E->trace.deltheta[0]),"1.0",m);
    input_double("regular_grid_delphi",&(E->trace.delphi[0]),"1.0",m);

    /* Analytical Test Function */

    E->trace.ianalytical_tracer_test=0;
    /* input_int("analytical_tracer_test",&(E->trace.ianalytical_tracer_test),
              "0,0,nomax",m); */
}


/************** LOST SOULS ***************************************************/
/*                                                                           */
/* This function is used to transport tracers to proper processor domains.   */
/* (MPI parallel)                                                            */
/* All of the tracers that were sent to rlater arrays are destined to another*/
/* cap and sent there. Then they are raised up or down for multiple z procs. */
/* isend[j][n]=number of tracers this processor cap is sending to cap n      */
/* ireceive[j][n]=number of tracers this processor cap receiving from cap n  */


void full_lost_souls(struct All_variables *E)
{
    /* This code works only if E->sphere.caps_per_proc==1 */
    const int j = 1;

    int ithiscap;
    int ithatcap=1;
    int isend[13][13];
    int ireceive[13][13];
    int isize[13];
    int kk,pp;
    int mm;
    int numtracers;
    int icheck=0;
    int isend_position;
    int ipos,ipos2,ipos3;
    int idb;
    int idestination_proc=0;
    int isource_proc;
    int isend_z[13][3];
    int ireceive_z[13][3];
    int isum[13];
    int irad;
    int ival;
    int ithat_processor;
    int ireceive_position;
    int ivertical_neighbor;
    int ilast_receiver_position;
    int it;
    int irec[13];
    int irec_position;
    int iel;
    int num_tracers;
    int isize_send;
    int isize_receive;
    int itemp_size;
    int itracers_subject_to_vertical_transport[13];

    double x,y,z;
    double theta,phi,rad;
    double *send[13][13];
    double *receive[13][13];
    double *send_z[13][3];
    double *receive_z[13][3];
    double *REC[13];

    double begin_time = CPU_time0();

    int number_of_caps=12;
    int lev=E->mesh.levmax;
    int num_ngb = E->parallel.TNUM_PASS[lev][j];
	
	Tracer temp_tracer;

    /* Note, if for some reason, the number of neighbors exceeds */
    /* 50, which is unlikely, the MPI arrays must be increased.  */
    MPI_Status status[200];
    MPI_Request request[200];
    MPI_Status status1;
    MPI_Status status2;
    int itag=1;


    parallel_process_sync(E);
    if(E->control.verbose)
      fprintf(E->trace.fpt, "Entering lost_souls()\n");


    E->trace.istat_isend=E->trace.escaped_tracers[j].size();
    /** debug **
    for (kk=1; kk<=E->trace.istat_isend; kk++) {
        fprintf(E->trace.fpt, "tracer#=%d xx=(%g,%g,%g)\n", kk,
                E->trace.rlater[j][0][kk],
                E->trace.rlater[j][1][kk],
                E->trace.rlater[j][2][kk]);
    }
    fflush(E->trace.fpt);
    /**/



    /* initialize isend and ireceive */
    /* # of neighbors in the horizontal plane */
    isize[j]=E->trace.escaped_tracers[j].size()*temp_tracer.size();
    for (kk=0;kk<=num_ngb;kk++) isend[j][kk]=0;
    for (kk=0;kk<=num_ngb;kk++) ireceive[j][kk]=0;

    /* Allocate Maximum Memory to Send Arrays */

    itemp_size=citmax(isize[j],1);

    for (kk=0;kk<=num_ngb;kk++) {
        if ((send[j][kk]=(double *)malloc(itemp_size*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"Error(lost souls)-no memory (u389)\n");
            fflush(E->trace.fpt);
            exit(10);
        }
    }


    /** debug **
    ithiscap=E->sphere.capid[j];
    for (kk=1;kk<=num_ngb;kk++) {
        ithatcap=E->parallel.PROCESSOR[lev][j].pass[kk];
        fprintf(E->trace.fpt,"cap: %d me %d TNUM: %d rank: %d\n",
                ithiscap,E->parallel.me,kk,ithatcap);

    }
    fflush(E->trace.fpt);
    /**/


    /* Pre communication */
    full_put_lost_tracers(E, isend, send);


    /* Send info to other processors regarding number of send tracers */

    /* idb is the request array index variable */
    /* Each send and receive has a request variable */

    idb=0;
    ithiscap=0;

    /* if tracer is in same cap (nprocz>1) */

    if (E->parallel.nprocz>1) {
        ireceive[j][ithiscap]=isend[j][ithiscap];
    }

    for (kk=1;kk<=num_ngb;kk++) {
        idestination_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

        MPI_Isend(&isend[j][kk],1,MPI_INT,idestination_proc,
                  11,E->parallel.world,&request[idb++]);

        MPI_Irecv(&ireceive[j][kk],1,MPI_INT,idestination_proc,
                  11,E->parallel.world,&request[idb++]);

    } /* end kk, number of neighbors */

    /* Wait for non-blocking calls to complete */

    MPI_Waitall(idb,request,status);


    /** debug **
    for (kk=0;kk<=num_ngb;kk++) {
        if(kk==0)
	  isource_proc=E->parallel.me;
        else
	  isource_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

	fprintf(E->trace.fpt,"%d send %d to proc %d\n",
		E->parallel.me,isend[j][kk],isource_proc);
	fprintf(E->trace.fpt,"%d recv %d from proc %d\n",
		E->parallel.me,ireceive[j][kk],isource_proc);
    }
    /**/

    /* Allocate memory in receive arrays */

    for (ithatcap=0;ithatcap<=num_ngb;ithatcap++) {
        isize[j]=ireceive[j][ithatcap]*temp_tracer.size();

        itemp_size=citmax(1,isize[j]);

        if ((receive[j][ithatcap]=(double *)malloc(itemp_size*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"Error(lost souls)-no memory (c721)\n");
            fflush(E->trace.fpt);
            exit(10);
        }
    }

    /* Now, send the tracers to proper caps */

    idb=0;
    ithiscap=0;

    /* same cap */

    if (E->parallel.nprocz>1) {

        ithatcap=ithiscap;
        isize[j]=isend[j][ithatcap]*temp_tracer.size();
        for (mm=0;mm<isize[j];mm++) {
            receive[j][ithatcap][mm]=send[j][ithatcap][mm];
        }

    }

    /* neighbor caps */

    for (kk=1;kk<=num_ngb;kk++) {
        idestination_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

        isize[j]=isend[j][kk]*temp_tracer.size();

        MPI_Isend(send[j][kk],isize[j],MPI_DOUBLE,idestination_proc,
                  11,E->parallel.world,&request[idb++]);

        isize[j]=ireceive[j][kk]*temp_tracer.size();

        MPI_Irecv(receive[j][kk],isize[j],MPI_DOUBLE,idestination_proc,
                  11,E->parallel.world,&request[idb++]);

    } /* end kk, number of neighbors */

    /* Wait for non-blocking calls to complete */

    MPI_Waitall(idb,request,status);


    /* Put all received tracers in array REC[j] */
    /* This makes things more convenient.       */

    /* Sum up size of receive arrays (all tracers sent to this processor) */

    isum[j]=0;

    ithiscap=0;

    for (kk=0;kk<=num_ngb;kk++) {
        isum[j]=isum[j]+ireceive[j][kk];
    }

    itracers_subject_to_vertical_transport[j]=isum[j];


    /* Allocate Memory for REC array */

    isize[j]=isum[j]*temp_tracer.size();
    isize[j]=citmax(isize[j],1);
    if ((REC[j]=(double *)malloc(isize[j]*sizeof(double)))==NULL) {
        fprintf(E->trace.fpt,"Error(lost souls)-no memory (g323)\n");
        fflush(E->trace.fpt);
        exit(10);
    }


    /* Put Received tracers in REC */
    irec[j]=0;

    irec_position=0;

    for (kk=0;kk<=num_ngb;kk++) {

        ithatcap=kk;

        for (pp=0;pp<ireceive[j][ithatcap];pp++) {
            irec[j]++;
            ipos=pp*temp_tracer.size();

            for (mm=0;mm<temp_tracer.size();mm++) {
                ipos2=ipos+mm;
                REC[j][irec_position]=receive[j][ithatcap][ipos2];

                irec_position++;

            } /* end mm (cycling tracer quantities) */
        } /* end pp (cycling tracers) */
    } /* end kk (cycling neighbors) */


    /* Done filling REC */


    /* VERTICAL COMMUNICATION */

    if (E->parallel.nprocz>1) {

        /* Allocate memory for send_z */
        /* Make send_z the size of receive array (max size) */
        /* (No dynamic reallocation of send_z necessary)    */

        for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++) {
            isize[j]=itracers_subject_to_vertical_transport[j]*temp_tracer.size();
            isize[j]=citmax(isize[j],1);

            if ((send_z[j][kk]=(double *)malloc(isize[j]*sizeof(double)))==NULL) {
                fprintf(E->trace.fpt,"Error(lost souls)-no memory (c721)\n");
                fflush(E->trace.fpt);
                exit(10);
            }
        }

        for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++) {

            ithat_processor=E->parallel.PROCESSORz[lev].pass[ivertical_neighbor];

            /* initialize isend_z and ireceive_z array */

            isend_z[j][ivertical_neighbor]=0;
            ireceive_z[j][ivertical_neighbor]=0;

            /* sort through receive array and check radius */

            it=0;
            num_tracers=irec[j];
            for (kk=1;kk<=num_tracers;kk++) {

                ireceive_position=it*temp_tracer.size();
                it++;

                irad=ireceive_position+2;

                rad=REC[j][irad];

                ival=icheck_that_processor_shell(E,j,ithat_processor,rad);


                /* if tracer is in other shell, take out of receive array and give to send_z*/

                if (ival==1) {


                    isend_position=isend_z[j][ivertical_neighbor]*temp_tracer.size();
                    isend_z[j][ivertical_neighbor]++;

                    ilast_receiver_position=(irec[j]-1)*temp_tracer.size();

                    for (mm=0;mm<temp_tracer.size();mm++) {
                        ipos=ireceive_position+mm;
                        ipos2=isend_position+mm;

                        send_z[j][ivertical_neighbor][ipos2]=REC[j][ipos];


                        /* eject tracer info from REC array, and replace with last tracer in array */

                        ipos3=ilast_receiver_position+mm;
                        REC[j][ipos]=REC[j][ipos3];

                    }

                    it--;
                    irec[j]--;

                } /* end if ival===1 */

                /* Otherwise, leave tracer */

            } /* end kk (cycling through tracers) */

        } /* end ivertical_neighbor */


        /* Send arrays are now filled.                         */
        /* Now send send information to vertical processor neighbor */
        idb=0;
        for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++) {

            idestination_proc = E->parallel.PROCESSORz[lev].pass[kk];
            MPI_Isend(&isend_z[j][kk],1,MPI_INT,idestination_proc,
                      14,E->parallel.world,&request[idb++]);

            MPI_Irecv(&ireceive_z[j][kk],1,MPI_INT,idestination_proc,
                      14,E->parallel.world,&request[idb++]);

        } /* end ivertical_neighbor */

        /* Wait for non-blocking calls to complete */

        MPI_Waitall(idb,request,status);


        /** debug **
        for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++) {
            fprintf(E->trace.fpt, "PROC: %d IVN: %d (P: %d) "
                    "SEND: %d REC: %d\n",
                    E->parallel.me,kk,E->parallel.PROCESSORz[lev].pass[kk],
                    isend_z[j][kk],ireceive_z[j][kk]);
        }
        fflush(E->trace.fpt);
        /**/


        /* Allocate memory to receive_z arrays */


        for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++) {
            isize[j]=ireceive_z[j][kk]*temp_tracer.size();
            isize[j]=citmax(isize[j],1);

            if ((receive_z[j][kk]=(double *)malloc(isize[j]*sizeof(double)))==NULL) {
                fprintf(E->trace.fpt,"Error(lost souls)-no memory (t590)\n");
                fflush(E->trace.fpt);
                exit(10);
            }
        }

        /* Send Tracers */

        idb=0;
        for (kk=1;kk<=E->parallel.TNUM_PASSz[lev];kk++) {

            idestination_proc = E->parallel.PROCESSORz[lev].pass[kk];

            isize_send=isend_z[j][kk]*temp_tracer.size();

            MPI_Isend(send_z[j][kk],isize_send,MPI_DOUBLE,idestination_proc,
                      15,E->parallel.world,&request[idb++]);

            isize_receive=ireceive_z[j][kk]*temp_tracer.size();

            MPI_Irecv(receive_z[j][kk],isize_receive,MPI_DOUBLE,idestination_proc,
                      15,E->parallel.world,&request[idb++]);
        }

        /* Wait for non-blocking calls to complete */

        MPI_Waitall(idb,request,status);


        /* Put tracers into REC array */

        /* First, reallocate memory to REC */

        isum[j]=0;
        for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++) {
            isum[j] += ireceive_z[j][ivertical_neighbor];
        }

        isum[j] += irec[j];
        isize[j]=isum[j]*temp_tracer.size();

        if (isize[j]>0) {
            if ((REC[j]=(double *)realloc(REC[j],isize[j]*sizeof(double)))==NULL) {
                fprintf(E->trace.fpt,"Error(lost souls)-no memory (i981)\n");
                fprintf(E->trace.fpt,"isize: %d\n",isize[j]);
                fflush(E->trace.fpt);
                exit(10);
            }
        }

        for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++) {

            for (kk=0;kk<ireceive_z[j][ivertical_neighbor];kk++) {

                irec_position=irec[j]*temp_tracer.size();
                irec[j]++;
                ireceive_position=kk*temp_tracer.size();

                for (mm=0;mm<temp_tracer.size();mm++) {
                    REC[j][irec_position+mm]=receive_z[j][ivertical_neighbor][ireceive_position+mm];
                }
            }

        }

        /* Free Vertical Arrays */
        for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++) {
            free(send_z[j][ivertical_neighbor]);
            free(receive_z[j][ivertical_neighbor]);
        }

    } /* endif nprocz>1 */

    /* END OF VERTICAL TRANSPORT */

    /* Put away tracers */


    for (kk=0;kk<irec[j];kk++) {
		Tracer new_tracer;
		CartesianCoord	cc;
		SphericalCoord	sc;

        ireceive_position=kk*new_tracer.size();
		
		new_tracer.readFromMem(&REC[j][ireceive_position]);

		sc = new_tracer.getSphericalPos();
		cc = new_tracer.getCartesianPos();
		
        iel=(E->trace.iget_element)(E,j,UNDEFINED_ELEMENT,cc,sc);

        if (iel<1) {
            fprintf(E->trace.fpt,"Error(lost souls) - element not here?\n");
            fprintf(E->trace.fpt,"x,y,z-theta,phi,rad: %f %f %f - %f %f %f\n",
					cc._x,cc._y,cc._z,sc._theta,sc._phi,sc._rad);
            fflush(E->trace.fpt);
            exit(10);
        }

        new_tracer.set_ielement(iel);
		E->trace.tracers[j].push_back(new_tracer);
    }
    if(E->control.verbose){
      fprintf(E->trace.fpt,"Freeing memory in lost_souls()\n");
      fflush(E->trace.fpt);
    }
    parallel_process_sync(E);

    /* Free Arrays */

    free(REC[j]);

    for (kk=0;kk<=num_ngb;kk++) {
        free(send[j][kk]);
        free(receive[j][kk]);

    }
    if(E->control.verbose){
      fprintf(E->trace.fpt,"Leaving lost_souls()\n");
      fflush(E->trace.fpt);
    }

    E->trace.lost_souls_time += CPU_time0() - begin_time;
}


static void full_put_lost_tracers(struct All_variables *E,
                                  int isend[13][13], double *send[13][13])
{
    const int j = 1;
    int kk, pp;
    int ithatcap, icheck;
    int isend_position, ipos;
    int lev = E->mesh.levmax;
	TracerList::iterator tr;
	CartesianCoord	cc;

    /* transfer tracers from rlater to send */

    for (tr=E->trace.escaped_tracers[j].begin();tr!=E->trace.escaped_tracers[j].end();++tr) {
		cc = tr->getCartesianPos();

        /* first check same cap if nprocz>1 */

        if (E->parallel.nprocz>1) {
            ithatcap=0;
            icheck=full_icheck_cap(E,ithatcap,cc,tr->rad());
            if (icheck==1) goto foundit;

        }

        /* check neighboring caps */

        for (pp=1;pp<=E->parallel.TNUM_PASS[lev][j];pp++) {
            ithatcap=pp;
            icheck=full_icheck_cap(E,ithatcap,cc,tr->rad());
            if (icheck==1) goto foundit;
        }


        /* should not be here */
        if (icheck!=1) {
            fprintf(E->trace.fpt,"Error(lost souls)-should not be here\n");
            fprintf(E->trace.fpt,"x: %f y: %f z: %f rad: %f\n",cc._x,cc._y,cc._z,tr->rad());
            icheck=full_icheck_cap(E,0,cc,tr->rad());
            if (icheck==1) fprintf(E->trace.fpt," icheck here!\n");
            else fprintf(E->trace.fpt,"icheck not here!\n");
            fflush(E->trace.fpt);
            exit(10);
        }

    foundit:

        isend[j][ithatcap]++;

        /* assign tracer to send */

        isend_position=(isend[j][ithatcap]-1)*tr->size();
		tr->writeToMem(&send[j][ithatcap][isend_position]);

    } /* end kk, assigning tracers */

	E->trace.escaped_tracers[j].clear();
}

/************************ GET SHAPE FUNCTION *********************************/
/* Real theta,phi,rad space is transformed into u,v space. This transformation */
/* maps great circles into straight lines. Here, elements boundaries are     */
/* assumed to be great circle planes (not entirely true, it is actually only */
/* the nodal arrangement that lies upon great circles).  Element boundaries  */
/* are then mapped into planes.  The element is then divided into 2 wedges   */
/* in which standard shape functions are used to interpolate velocity.       */
/* This transformation was found on the internet (refs were difficult to     */
/* to obtain). It was tested that nodal configuration is indeed transformed  */
/* into straight lines.                                                      */
/* Radial and azimuthal shape functions are decoupled. First find the shape  */
/* functions associated with the 2D surface plane, then apply radial shape   */
/* functions.                                                                */
/*                                                                           */
/* Wedge information:                                                        */
/*                                                                           */
/*        Wedge 1                  Wedge 2                                   */
/*        _______                  _______                                   */
/*                                                                           */
/*    wedge_node  real_node      wedge_node  real_node                       */
/*    ----------  ---------      ----------  ---------                       */
/*                                                                           */
/*         1        1               1            1                           */
/*         2        2               2            3                           */
/*         3        3               3            4                           */
/*         4        5               4            5                           */
/*         5        6               5            7                           */
/*         6        7               6            8                           */

void full_get_shape_functions(struct All_variables *E, int &shape_iwedge,
                              double shp[6], ElementID nelem,
                              SphericalCoord sc)
{
    const int j = 1;

    int iwedge,inum;
    int i, kk;
    int ival;
    int itry;

	CoordUV		uv;
    double shape2d[3];
    double shaperad[2];
    double shape[7];

    int maxlevel=E->mesh.levmax;

    const double eps=-1e-4;

    /* find u and v using spherical coordinates */

    uv = spherical_to_uv(E,sc);

    inum=0;
    itry=1;

 try_again:

    /* Check first wedge (1 of 2) */

    iwedge=0;

 next_wedge:

    /* determine shape functions of wedge */
    /* There are 3 shape functions for the triangular wedge */

    get_2dshape(E,j,nelem,uv,iwedge,shape2d);

    /* if any shape functions are negative, goto next wedge */

    if (shape2d[0]<eps||shape2d[1]<eps||shape2d[2]<eps)
        {
            inum++;
            /* AKMA clean this up */
            if (inum>3)
                {
                    fprintf(E->trace.fpt,"ERROR(gnomonic_interpolation)-inum>3!\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            if (inum>1 && itry==1)
                {
					CartesianCoord	cc;
                    fprintf(E->trace.fpt,"ERROR(gnomonic_interpolation)-inum>1\n");
                    fprintf(E->trace.fpt,"shape %f %f %f\n",shape2d[0],shape2d[1],shape2d[2]);
                    fprintf(E->trace.fpt,"u %f v %f element: %d \n",uv.u,uv.v, nelem);
                    fprintf(E->trace.fpt,"Element uv boundaries: \n");
                    for(kk=1;kk<=4;kk++) {
                        i = (E->ien[j][nelem].node[kk] - 1) / E->lmesh.noz + 1;
                        fprintf(E->trace.fpt,"%d: U: %f V:%f\n",kk,(*E->trace.gnomonic)[i].u,(*E->trace.gnomonic)[i].v);
                    }
                    fprintf(E->trace.fpt,"theta: %f phi: %f rad: %f\n",sc._theta,sc._phi,sc._rad);
                    fprintf(E->trace.fpt,"Element theta-phi boundaries: \n");
                    for(kk=1;kk<=4;kk++)
                        fprintf(E->trace.fpt,"%d: Theta: %f Phi:%f\n",kk,E->sx[j][1][E->ien[j][nelem].node[kk]],E->sx[j][2][E->ien[j][nelem].node[kk]]);
                    cc = sc.toCartesian();

                    ival=icheck_element(E,j,nelem,cc,sc._rad);
                    fprintf(E->trace.fpt,"ICHECK?: %d\n",ival);
					
                    ival=(E->trace.iget_element)(E, j, UNDEFINED_ELEMENT, cc, sc);
                    fprintf(E->trace.fpt,"New Element?: %d\n",ival);
					
                    ival=icheck_column_neighbors(E,j,nelem,cc,sc._rad);
                    fprintf(E->trace.fpt,"New Element (neighs)?: %d\n",ival);
					
                    nelem=ival;
                    ival=icheck_element(E,j,nelem,cc,sc._rad);
                    fprintf(E->trace.fpt,"ICHECK?: %d\n",ival);
					
                    itry++;
                    if (ival>0) goto try_again;
                    fprintf(E->trace.fpt,"NO LUCK\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

            iwedge=1;
            goto next_wedge;
        }

    /* Determine radial shape functions */
    /* There are 2 shape functions radially */

    get_radial_shape(E,j,nelem,sc._rad,shaperad);

    /* There are 6 nodes to the solid wedge.             */
    /* The 6 shape functions assocated with the 6 nodes  */
    /* are products of radial and wedge shape functions. */

    /* Sum of shape functions is 1                       */

    shape_iwedge = iwedge;
    shp[0]=shaperad[0]*shape2d[0];
    shp[1]=shaperad[0]*shape2d[1];
    shp[2]=shaperad[0]*shape2d[2];
    shp[3]=shaperad[1]*shape2d[0];
    shp[4]=shaperad[1]*shape2d[1];
    shp[5]=shaperad[1]*shape2d[2];

    /** debug **
    fprintf(E->trace.fpt, "shp: %e %e %e %e %e %e\n",
            shp[0], shp[1], shp[2], shp[3], shp[4], shp[5]);
    /**/
}


/************************ GET VELOCITY ***************************************/
/*                                                                           */
/* This function interpolates tracer velocity using gnominic interpolation.  */
/* The element is divided into 2 wedges in which standard shape functions    */
/* are used to interpolate velocity.                                         */
/*                                                                           */
/* Wedge information:                                                        */
/*                                                                           */
/*        Wedge 1                  Wedge 2                                   */
/*        _______                  _______                                   */
/*                                                                           */
/*    wedge_node  real_node      wedge_node  real_node                       */
/*    ----------  ---------      ----------  ---------                       */
/*                                                                           */
/*         1        1               1            1                           */
/*         2        2               2            3                           */
/*         3        3               3            4                           */
/*         4        5               4            5                           */
/*         5        6               5            7                           */
/*         6        7               6            8                           */

CartesianCoord full_get_velocity(struct All_variables *E,
								 int j, ElementID nelem,
								 SphericalCoord sc)
{
    int iwedge, i;
    const int sphere_key = 0;
	
    double			shape[6];
    CartesianCoord	VV[9], vel[6], result;
	
    full_get_shape_functions(E, iwedge, shape, nelem, sc);
	
    /* get cartesian velocity */
    velo_from_element_d(E, VV, j, nelem, sphere_key);
	
    /* depending on wedge, set up velocity points */
	
    if (iwedge==0) {
		vel[0]=VV[1];
		vel[1]=VV[2];
		vel[2]=VV[3];
		vel[3]=VV[5];
		vel[4]=VV[6];
		vel[5]=VV[7];
	} else if (iwedge==1) {
		vel[0]=VV[1];
		vel[1]=VV[3];
		vel[2]=VV[4];
		vel[3]=VV[5];
		vel[4]=VV[7];
		vel[5]=VV[8];
	}
	
	for (i=0;i<6;++i) result += vel[i] * shape[i];
	return result;
}

/***************************************************************/
/* GET 2DSHAPE                                                 */
/*                                                             */
/* This function determines shape functions at u,v             */
/* This method uses standard linear shape functions of         */
/* triangular elements. (See Cuvelier, Segal, and              */
/* van Steenhoven, 1986).                                      */

static void get_2dshape(struct All_variables *E,
                        int j, ElementID nelem,
                        CoordUV uv,
                        int iwedge, double * shape2d)
{
    /* convert nelem to surface element number */
    int n = (nelem - 1) / E->lmesh.elz + 1;

    /* shape functions */
	shape2d[0] = E->trace.shape_coefs[j][iwedge][n]->applyShapeFunc(0, uv);
	shape2d[1] = E->trace.shape_coefs[j][iwedge][n]->applyShapeFunc(1, uv);
	shape2d[2] = E->trace.shape_coefs[j][iwedge][n]->applyShapeFunc(2, uv);

    /** debug **
    fprintf(E->trace.fpt, "el=%d els=%d iwedge=%d shape=(%e %e %e)\n",
            nelem, n, iwedge, shape2d[0], shape2d[1], shape2d[2]);
    /**/
}

/***************************************************************/
/* GET RADIAL SHAPE                                            */
/*                                                             */
/* This function determines radial shape functions at rad      */

static void get_radial_shape(struct All_variables *E,
                             int j, ElementID nelem,
                             double rad, double *shaperad)
{

    int node1,node5;
    double rad1,rad5,f1,f2,delrad;
    const double eps=1e-6;
    double top_bound=1.0+eps;
    double bottom_bound=0.0-eps;

    node1=E->ien[j][nelem].node[1];
    node5=E->ien[j][nelem].node[5];

    rad1=E->sx[j][3][node1];
    rad5=E->sx[j][3][node5];

    delrad=rad5-rad1;

    f1=(rad-rad1)/delrad;
    f2=(rad5-rad)/delrad;

    /* Save a small amount of computation here   */
    /* because f1+f2=1, shapes can be switched   */
    /*
      shaperad[0]=1.0-f1=1.0-(1.0-f2)=f2;
      shaperad[1]=1.0-f2=1.0-(10-f1)=f1;
    */

    shaperad[0]=f2;
    shaperad[1]=f1;

    /* Some error control */

    if (shaperad[0]>top_bound || shaperad[0]<bottom_bound ||
        shaperad[1]>top_bound || shaperad[1]<bottom_bound)
        {
            fprintf(E->trace.fpt,"ERROR(get_radial_shape)\n");
            fprintf(E->trace.fpt,"shaperad[0]: %f \n",shaperad[0]);
            fprintf(E->trace.fpt,"shaperad[1]: %f \n",shaperad[1]);
            fflush(E->trace.fpt);
            exit(10);
        }
}


/**************************************************************/
/* SPHERICAL TO UV                                               */
/*                                                            */
/* This function transforms theta and phi to new coords       */
/* u and v using gnomonic projection.                          */

static CoordUV spherical_to_uv(struct All_variables *E,
                            SphericalCoord sc)
{
    double phi_f;
    double cosc;
    double cos_theta_f,sin_theta_f;
    double cost,sint,cosp2,sinp2;

    /* theta_f and phi_f are the reference points of the cap */

    phi_f = E->trace.gnomonic_reference_phi;

    cos_theta_f = (*E->trace.gnomonic)[0].u;
    sin_theta_f = (*E->trace.gnomonic)[0].v;

    cost=cos(sc._theta);
    /*
      sint=sin(theta);
    */
    sint=sqrt(1.0-cost*cost);

    cosp2=cos(sc._phi-phi_f);
    sinp2=sin(sc._phi-phi_f);

    cosc=cos_theta_f*cost+sin_theta_f*sint*cosp2;
    cosc=1.0/cosc;
	
	return CoordUV(sint*sinp2*cosc, (sin_theta_f*cost-cos_theta_f*sint*cosp2)*cosc);

    /** debug **
    fprintf(E->trace.fpt, "(%e %e) -> (%e %e)\n",
            theta, phi, *u, *v);
    /**/
}


/*********** MAKE REGULAR GRID ********************************/
/*                                                            */
/* This function generates the finer regular grid which is    */
/* mapped to real elements                                    */

void make_regular_grid(struct All_variables *E)
{
    int j;
    int kk;
    int mm;
    int pp,node;
    int numtheta,numphi;
    int nodestheta,nodesphi;
    unsigned int numregel;
    unsigned int numregnodes;
    int idum1,idum2;
    int ifound_one;
    int ival;
    int ilast_el;
    int imap;
    int elz;
    int nelsurf;
    int iregnode[5];
    int ntheta,nphi;
    int ichoice;
    int icount;
    int itemp[5];
    int iregel;
    int istat_ichoice[13][5];
    int isum;
	
    double x,y,z;
    double theta,phi,rad;
    double deltheta;
    double delphi;
    double thetamax,thetamin;
    double phimax,phimin;
    double start_time;
    double theta_min,phi_min;
    double theta_max,phi_max;
    double half_diff;
    double expansion;
	
    double *tmin;
    double *tmax;
    double *fmin;
    double *fmax;
	
    const double two_pi=2.0*M_PI;
	
    elz=E->lmesh.elz;
	
    nelsurf=E->lmesh.elx*E->lmesh.ely;
	
    //TODO: find the bounding box of the mesh, if the box is too close to
    // to core, set a flag (rotated_reggrid) to true and rotate the
    // bounding box to the equator. Generate the regular grid with the new
    // bounding box. The rotation should be a simple one, e.g.
    // (theta, phi) -> (??)
    // Whenever the regular grid is used, check the flat (rotated_reggrid),
    // if true, rotate the checkpoint as well.
	
    /* note, mesh is rotated along theta 22.5 degrees divided by elx. */
    /* We at least want that much expansion here! Otherwise, theta min */
    /* will not be valid near poles. We do a little more (x2) to be safe */
    /* Typically 1-2 degrees. Look in nodal_mesh.c for this.             */
	
    expansion=2.0*0.5*(M_PI/4.0)/(1.0*E->lmesh.elx);
	
    start_time=CPU_time0();
	
    if (E->parallel.me==0) fprintf(stderr,"Generating Regular Grid\n");
	
	
    /* for each cap, determine theta and phi bounds, watch out near poles  */
	
    numregnodes=0;
    for(j=1;j<=E->sphere.caps_per_proc;j++)
	{
		
		thetamax=0.0;
		thetamin=M_PI;
		
		phimax=two_pi;
		phimin=0.0;
		
		for (kk=1;kk<=E->lmesh.nno;kk=kk+E->lmesh.noz)
		{
			
			theta=E->sx[j][1][kk];
			phi=E->sx[j][2][kk];
			
			thetamax=citmax(thetamax,theta);
			thetamin=citmin(thetamin,theta);
			
		}
		
		/* expand range slightly (should take care of poles)  */
		
		thetamax=thetamax+expansion;
		thetamax=citmin(thetamax,M_PI);
		
		thetamin=thetamin-expansion;
		thetamin=citmax(thetamin,0.0);
		
		/* Convert input data from degrees to radians  */
		
		deltheta=E->trace.deltheta[0]*M_PI/180.0;
		delphi=E->trace.delphi[0]*M_PI/180.0;
		
		
		/* Adjust deltheta and delphi to fit a uniform number of regular elements */
		
		numtheta=fabs(thetamax-thetamin)/deltheta;
		numphi=fabs(phimax-phimin)/delphi;
		nodestheta=numtheta+1;
		nodesphi=numphi+1;
		numregel=numtheta*numphi;
		numregnodes=nodestheta*nodesphi;
		
		if ((numtheta==0)||(numphi==0))
		{
			fprintf(E->trace.fpt,"Error(make_regular_grid): numtheta: %d numphi: %d\n",numtheta,numphi);
			fflush(E->trace.fpt);
			exit(10);
		}
		
		deltheta=fabs(thetamax-thetamin)/(1.0*numtheta);
		delphi=fabs(phimax-phimin)/(1.0*numphi);
		
		/* fill global variables */
		
		E->trace.deltheta[j]=deltheta;
		E->trace.delphi[j]=delphi;
		E->trace.numtheta[j]=numtheta;
		E->trace.numphi[j]=numphi;
		E->trace.thetamax[j]=thetamax;
		E->trace.thetamin[j]=thetamin;
		E->trace.phimax[j]=phimax;
		E->trace.phimin[j]=phimin;
		E->trace.numregel[j]=numregel;
		E->trace.numregnodes[j]=numregnodes;
		
		if ( ((1.0*numregel)/(1.0*E->lmesh.elx*E->lmesh.ely)) < 0.5 )
		{
			fprintf(E->trace.fpt,"\n ! WARNING: regular/real ratio low: %f ! \n",
					((1.0*numregel)/(1.0*E->lmesh.nel)) );
			fprintf(E->trace.fpt," Should reduce size of regular mesh\n");
			fprintf(stderr,"! WARNING: regular/real ratio low: %f ! \n",
					((1.0*numregel)/(1.0*E->lmesh.nel)) );
			fprintf(stderr," Should reduce size of regular mesh\n");
			fflush(E->trace.fpt);
			if (E->trace.itracer_warnings) exit(10);
		}
		
		/* print some output */
		
		fprintf(E->trace.fpt,"\nRegular grid:\n");
		fprintf(E->trace.fpt,"Theta min: %f max: %f \n",thetamin,thetamax);
		fprintf(E->trace.fpt,"Phi min: %f max: %f \n",phimin,phimax);
		fprintf(E->trace.fpt,"Adjusted deltheta: %f delphi: %f\n",deltheta,delphi);
		fprintf(E->trace.fpt,"(numtheta: %d  numphi: %d)\n",numtheta,numphi);
		fprintf(E->trace.fpt,"Number of regular elements: %d  (nodes: %d)\n",numregel,numregnodes);
		
		fprintf(E->trace.fpt,"regular/real ratio: %f\n",((1.0*numregel)/(1.0*E->lmesh.elx*E->lmesh.ely)));
		fflush(E->trace.fpt);
		
		/* Allocate memory for regnodetoel */
		/* Regtoel is an integer array which represents nodes on    */
		/* the regular mesh. Each node on the regular mesh contains */
		/* the real element value if one exists (UNDEFINED_ELEMENT otherwise)     */
		
		
		
		if ((E->trace.regnodetoel[j]=(int *)malloc((numregnodes+1)*sizeof(int)))==NULL)
		{
			fprintf(E->trace.fpt,"ERROR(make regular) -no memory - uh3ud\n");
			fflush(E->trace.fpt);
			exit(10);
		}
		
		
		/* Initialize regnodetoel - reg elements not used = UNDEFINED_ELEMENT */
		
		for (kk=1;kk<=numregnodes;kk++)
		{
			E->trace.regnodetoel[j][kk]=UNDEFINED_ELEMENT;
		}
		
		/* Begin Mapping (only need to use surface elements) */
		
		parallel_process_sync(E);
		if (E->parallel.me==0) fprintf(stderr,"Beginning Mapping\n");
		
		/* Generate temporary arrays of max and min values for each surface element */
		
		
		if ((tmin=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
		{
			fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
			fflush(E->trace.fpt);
			exit(10);
		}
		if ((tmax=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
		{
			fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
			fflush(E->trace.fpt);
			exit(10);
		}
		if ((fmin=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
		{
			fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
			fflush(E->trace.fpt);
			exit(10);
		}
		if ((fmax=(double *)malloc((nelsurf+1)*sizeof(double)))==NULL)
		{
			fprintf(E->trace.fpt,"ERROR(make regular) -no memory - 7t1a\n");
			fflush(E->trace.fpt);
			exit(10);
		}
		
		for (mm=elz;mm<=E->lmesh.nel;mm=mm+elz)
		{
			
			kk=mm/elz;
			
			theta_min=M_PI;
			theta_max=0.0;
			phi_min=two_pi;
			phi_max=0.0;
			for (pp=1;pp<=4;pp++)
			{
				node=E->ien[j][mm].node[pp];
				theta=E->sx[j][1][node];
				phi=E->sx[j][2][node];
				
				theta_min=citmin(theta_min,theta);
				theta_max=citmax(theta_max,theta);
				phi_min=citmin(phi_min,phi);
				phi_max=citmax(phi_max,phi);
			}
			
			/* add half difference to phi and expansion to theta to be safe */
			
			theta_max=theta_max+expansion;
			theta_min=theta_min-expansion;
			
			theta_max=citmin(M_PI,theta_max);
			theta_min=citmax(0.0,theta_min);
			
			half_diff=0.5*(phi_max-phi_min);
			phi_max=phi_max+half_diff;
			phi_min=phi_min-half_diff;
			
			fix_angle(&phi_max);
			fix_angle(&phi_min);
			
			if (phi_min>phi_max)
			{
				phi_min=0.0;
				phi_max=two_pi;
			}
			
			tmin[kk]=theta_min;
			tmax[kk]=theta_max;
			fmin[kk]=phi_min;
			fmax[kk]=phi_max;
		}
		
		/* end looking through elements */
		
		ifound_one=0;
		
		rad=E->sphere.ro;
		
		imap=0;
		
		for (kk=1;kk<=numregnodes;kk++)
		{
			E->trace.regnodetoel[j][kk]=UNDEFINED_ELEMENT;
			
			/* find theta and phi for a given regular node */
			
			idum1=(kk-1)/(numtheta+1);
			idum2=kk-1-idum1*(numtheta+1);
			
			theta=thetamin+(1.0*idum2*deltheta);
			phi=phimin+(1.0*idum1*delphi);
			
			SphericalCoord		sc(theta,phi,rad);
			CartesianCoord		cc;
			
			cc = sc.toCartesian();
			
			ilast_el=1;
			
			/* if previous element not found yet, check all surface elements */
			
			/*
			 if (ifound_one==0)
			 {
			 for (mm=elz;mm<=E->lmesh.nel;mm=mm+elz)
			 {
			 pp=mm/elz;
			 if ( (theta>=tmin[pp]) && (theta<=tmax[pp]) && (phi>=fmin[pp]) && (phi<=fmax[pp]) )
			 {
			 ival=icheck_element_column(E,j,mm,x,y,z,rad);
			 if (ival>0)
			 {
			 ilast_el=mm;
			 ifound_one++;
			 E->trace.regnodetoel[j][kk]=mm;
			 goto foundit;
			 }
			 }
			 }
			 goto foundit;
			 }
			 */
			
			/* first check previous element */
			
			ival=icheck_element_column(E,j,ilast_el,cc,rad);
			if (ival>0)
			{
				E->trace.regnodetoel[j][kk]=ilast_el;
				goto foundit;
			}
			
			/* check neighbors */
			
			ival=icheck_column_neighbors(E,j,ilast_el,cc,rad);
			if (ival>0)
			{
				E->trace.regnodetoel[j][kk]=ival;
				ilast_el=ival;
				goto foundit;
			}
			
			/* check all */
			
			for (mm=elz;mm<=E->lmesh.nel;mm=mm+elz)
			{
				pp=mm/elz;
				if ( (theta>=tmin[pp]) && (theta<=tmax[pp]) && (phi>=fmin[pp]) && (phi<=fmax[pp]) )
				{
					ival=icheck_element_column(E,j,mm,cc,rad);
					if (ival>0)
					{
						ilast_el=mm;
						E->trace.regnodetoel[j][kk]=mm;
						goto foundit;
					}
				}
			}
			
		foundit:
			
			if (E->trace.regnodetoel[j][kk]>0) imap++;
			
		} /* end all regular nodes (kk) */
		
		fprintf(E->trace.fpt,"percentage mapped: %f\n", (1.0*imap)/(1.0*numregnodes)*100.0);
		fflush(E->trace.fpt);
		
		/* free temporary arrays */
		
		free(tmin);
		free(tmax);
		free(fmin);
		free(fmax);
		
	} /* end j */
	
	
    /* some error control */
	
    for (j=1;j<=E->sphere.caps_per_proc;j++)
	{
		for (kk=1;kk<=numregnodes;kk++)
		{
			if (E->trace.regnodetoel[j][kk]!=UNDEFINED_ELEMENT)
			{
				if ( (E->trace.regnodetoel[j][kk]<1)||(E->trace.regnodetoel[j][kk]>E->lmesh.nel) )
				{
					fprintf(stderr,"Error(make_regular_grid)-invalid element: %d\n",E->trace.regnodetoel[j][kk]);
					fprintf(E->trace.fpt,"Error(make_regular_grid)-invalid element: %d\n",E->trace.regnodetoel[j][kk]);
					fflush(E->trace.fpt);
					fflush(stderr);
					exit(10);
				}
			}
		}
	}
	
	
    /* Now put regnodetoel information into regtoel */
	
	
    if (E->parallel.me==0) fprintf(stderr,"Beginning Regtoel submapping \n");
	
    /* AKMA decided it would be more efficient to have reg element choice array */
    /* rather than reg node array as used before          */
	
	
    for(j=1;j<=E->sphere.caps_per_proc;j++)
	{
		
		/* initialize statistical counter */
		
		for (pp=0;pp<=4;pp++) istat_ichoice[j][pp]=0;
		
		/* Allocate memory for regtoel */
		/* Regtoel consists of 4 positions for each regular element */
		/* Position[0] lists the number of element choices (later   */
		/* referred to as ichoice), followed                        */
		/* by the possible element choices.                          */
		/* ex.) A regular element has 4 nodes. Each node resides in  */
		/* a real element. The number of real elements a regular     */
		/* element touches (one of its nodes are in) is ichoice.     */
		/* Special ichoice notes:                                    */
		/*    ichoice=-1   all regular element nodes = UNDEFINED_ELEMENT (no elements) */
		/*    ichoice=0    all 4 corners within one element              */
		/*    ichoice=1     one element choice (diff from ichoice 0 in    */
		/*                  that perhaps one reg node is in an element    */
		/*                  and the rest are not (UNDEFINED_ELEMENT).     */
		/*    ichoice>1     Multiple elements to check                    */
		
		numregel= E->trace.numregel[j];
		
		for (pp=0;pp<=4;pp++)
		{
			if ((E->trace.regtoel[j][pp]=(int *)malloc((numregel+1)*sizeof(int)))==NULL)
			{
				fprintf(E->trace.fpt,"ERROR(make regular)-no memory 98d (%d %d %d)\n",pp,numregel,j);
				fflush(E->trace.fpt);
				exit(10);
			}
		}
		
		numtheta=E->trace.numtheta[j];
		numphi=E->trace.numphi[j];
		
		for (nphi=1;nphi<=numphi;nphi++)
		{
			for (ntheta=1;ntheta<=numtheta;ntheta++)
			{
				
				iregel=ntheta+(nphi-1)*numtheta;
				
				/* initialize regtoel (not necessary really) */
				
				for (pp=0;pp<=4;pp++) E->trace.regtoel[j][pp][iregel]=-33;
				
				if ( (iregel>numregel)||(iregel<1) )
				{
					fprintf(E->trace.fpt,"ERROR(make_regular_grid)-weird iregel: %d (max: %d)\n",iregel,numregel);
					fflush(E->trace.fpt);
					exit(10);
				}
				
				iregnode[1]=iregel+(nphi-1);
				iregnode[2]=iregel+nphi;
				iregnode[3]=iregel+nphi+E->trace.numtheta[j]+1;
				iregnode[4]=iregel+nphi+E->trace.numtheta[j];
				
				for (kk=1;kk<=4;kk++)
				{
					if ((iregnode[kk]<1)||(iregnode[kk]>numregnodes))
					{
						fprintf(E->trace.fpt,"ERROR(make regular)-bad regnode %d\n",iregnode[kk]);
						fflush(E->trace.fpt);
						exit(10);
					}
					if (E->trace.regnodetoel[j][iregnode[kk]]>E->lmesh.nel)
					{
						fprintf(E->trace.fpt,"AABB HERE %d %d %d %d\n",iregel,iregnode[kk],kk,E->trace.regnodetoel[j][iregnode[kk]]);
						fflush(E->trace.fpt);
					}
				}
				
				
				/* find number of choices */
				
				ichoice=0;
				icount=0;
				
				for (kk=1;kk<=4;kk++)
				{
					
					if (E->trace.regnodetoel[j][iregnode[kk]]<=0) goto next_corner;
					
					icount++;
					for (pp=1;pp<=(kk-1);pp++)
					{
						if (E->trace.regnodetoel[j][iregnode[kk]]==E->trace.regnodetoel[j][iregnode[pp]]) goto next_corner;
					}
					ichoice++;
					itemp[ichoice]=E->trace.regnodetoel[j][iregnode[kk]];
					
					if ((ichoice<0) || (ichoice>4) )
					{
						fprintf(E->trace.fpt,"ERROR(make regular) - weird ichoice %d \n",ichoice);
						fflush(E->trace.fpt);
						exit(10);
					}
					if ((itemp[ichoice]<0) || (itemp[ichoice]>E->lmesh.nel) )
					{
						fprintf(E->trace.fpt,"ERROR(make regular) - weird element choice %d %d\n",itemp[ichoice],ichoice);
						fflush(E->trace.fpt);
						exit(10);
					}
					
				next_corner:
					;
				} /* end kk */
				
				istat_ichoice[j][ichoice]++;
				
				if ((ichoice<0) || (ichoice>4))
				{
					fprintf(E->trace.fpt,"ERROR(make_regular)-wierd ichoice %d\n",ichoice);
					fflush(E->trace.fpt);
					exit(10);
				}
				
				if (ichoice==0)
				{
					E->trace.regtoel[j][0][iregel]=-1;
					/*
					 fprintf(E->trace.fpt,"HH1: (%p) iregel: %d ichoice: %d value: %d %d\n",&E->trace.regtoel[j][1][iregel],iregel,ichoice,E->trace.regtoel[j][0][iregel],E->trace.regtoel[j][1][iregel]);
					 */
				}
				else if ( (ichoice==1) && (icount==4) )
				{
					E->trace.regtoel[j][0][iregel]=0;
					E->trace.regtoel[j][1][iregel]=itemp[1];
					
					/*
					 fprintf(E->trace.fpt,"HH2: (%p) iregel: %d ichoice: %d value: %d %d\n",&E->trace.regtoel[j][1][iregel],iregel,ichoice,E->trace.regtoel[j][0][iregel],E->trace.regtoel[j][1][iregel]);
					 */
					
					if (itemp[1]<1 || itemp[1]>E->lmesh.nel)
					{
						fprintf(E->trace.fpt,"ERROR(make_regular)-huh? wierd itemp\n");
						fflush(E->trace.fpt);
						exit(10);
					}
				}
				else if ( (ichoice>0) && (ichoice<5) )
				{
					E->trace.regtoel[j][0][iregel]=ichoice;
					for (pp=1;pp<=ichoice;pp++)
					{
						E->trace.regtoel[j][pp][iregel]=itemp[pp];
						
						/*
						 fprintf(E->trace.fpt,"HH:(%p)  iregel: %d ichoice: %d pp: %d value: %d %d\n",&E->trace.regtoel[j][pp][iregel],iregel,ichoice,pp,itemp[pp],E->trace.regtoel[j][pp][iregel]);
						 */
						if (itemp[pp]<1 || itemp[pp]>E->lmesh.nel)
						{
							fprintf(E->trace.fpt,"ERROR(make_regular)-huh? wierd itemp 2 \n");
							fflush(E->trace.fpt);
							exit(10);
						}
					}
				}
				else
				{
					fprintf(E->trace.fpt,"ERROR(make_regular)- should not be here! %d\n",ichoice);
					fflush(E->trace.fpt);
					exit(10);
				}
			}
		}
		
		/* can now free regnodetoel */
		
		free (E->trace.regnodetoel[j]);
		
		
		/* testing */
		for (kk=1;kk<=E->trace.numregel[j];kk++)
		{
			if ((E->trace.regtoel[j][0][kk]<-1)||(E->trace.regtoel[j][0][kk]>4))
			{
				fprintf(E->trace.fpt,"ERROR(make regular) regtoel ichoice0? %d %d \n",kk,E->trace.regtoel[j][pp][kk]);
				fflush(E->trace.fpt);
				exit(10);
			}
			for (pp=1;pp<=4;pp++)
			{
				if (((E->trace.regtoel[j][pp][kk]<1)&&(E->trace.regtoel[j][pp][kk]!=-33))||(E->trace.regtoel[j][pp][kk]>E->lmesh.nel))
				{
					fprintf(E->trace.fpt,"ERROR(make regular) (%p) regtoel? %d %d(%d) %d\n",&E->trace.regtoel[j][pp][kk],kk,pp,E->trace.regtoel[j][0][kk],E->trace.regtoel[j][pp][kk]);
					fflush(E->trace.fpt);
					exit(10);
				}
			}
		}
		
	} /* end j */
	
	
    fprintf(E->trace.fpt,"Mapping completed (%f seconds)\n",CPU_time0()-start_time);
    fflush(E->trace.fpt);
	
    parallel_process_sync(E);
	
    if (E->parallel.me==0) fprintf(stderr,"Mapping completed (%f seconds)\n",CPU_time0()-start_time);
	
    /* Print out information regarding regular/real element coverage */
	
    for (j=1;j<=E->sphere.caps_per_proc;j++)
	{
		isum=0;
		for (kk=0;kk<=4;kk++) isum=isum+istat_ichoice[j][kk];
		fprintf(E->trace.fpt,"\n\nInformation regarding number of real elements per regular elements\n");
		fprintf(E->trace.fpt," (stats done on regular elements that were used)\n");
		fprintf(E->trace.fpt,"Ichoice is number of real elements touched by a regular element\n");
		fprintf(E->trace.fpt,"  (ichoice=0 is optimal)\n");
		fprintf(E->trace.fpt,"Ichoice=0: %f percent\n",(100.0*istat_ichoice[j][0])/(1.0*isum));
		fprintf(E->trace.fpt,"Ichoice=1: %f percent\n",(100.0*istat_ichoice[j][1])/(1.0*isum));
		fprintf(E->trace.fpt,"Ichoice=2: %f percent\n",(100.0*istat_ichoice[j][2])/(1.0*isum));
		fprintf(E->trace.fpt,"Ichoice=3: %f percent\n",(100.0*istat_ichoice[j][3])/(1.0*isum));
		fprintf(E->trace.fpt,"Ichoice=4: %f percent\n",(100.0*istat_ichoice[j][4])/(1.0*isum));
		
	} /* end j */
}


/*********  ICHECK COLUMN NEIGHBORS ***************************/
/*                                                            */
/* This function check whether a point is in a neighboring    */
/* column. Neighbor surface element number is returned        */

static int icheck_column_neighbors(struct All_variables *E,
                                   int j, ElementID nel,
                                   CartesianCoord cc,
                                   double rad)
{
    int ival;
    int neighbor[25];
    int elx,ely,elz;
    int elxz;
    int kk;
	
    /*
	 const int number_of_neighbors=24;
	 */
	
    /* maybe faster to only check inner ring */
	
    const int number_of_neighbors=8;
	
    elx=E->lmesh.elx;
    ely=E->lmesh.ely;
    elz=E->lmesh.elz;
	
    elxz=elx*elz;
	
    /* inner ring */
	
    neighbor[1]=nel-elxz-elz;
    neighbor[2]=nel-elxz;
    neighbor[3]=nel-elxz+elz;
    neighbor[4]=nel-elz;
    neighbor[5]=nel+elz;
    neighbor[6]=nel+elxz-elz;
    neighbor[7]=nel+elxz;
    neighbor[8]=nel+elxz+elz;
	
    /* outer ring */
	
    neighbor[9]=nel+2*elxz-elz;
    neighbor[10]=nel+2*elxz;
    neighbor[11]=nel+2*elxz+elz;
    neighbor[12]=nel+2*elxz+2*elz;
    neighbor[13]=nel+elxz+2*elz;
    neighbor[14]=nel+2*elz;
    neighbor[15]=nel-elxz+2*elz;
    neighbor[16]=nel-2*elxz+2*elz;
    neighbor[17]=nel-2*elxz+elz;
    neighbor[18]=nel-2*elxz;
    neighbor[19]=nel-2*elxz-elz;
    neighbor[20]=nel-2*elxz-2*elz;
    neighbor[21]=nel-elxz-2*elz;
    neighbor[22]=nel-2*elz;
    neighbor[23]=nel+elxz-2*elz;
    neighbor[24]=nel+2*elxz-2*elz;
	
    for (kk=1;kk<=number_of_neighbors;kk++)
	{
		if ((neighbor[kk]>=1)&&(neighbor[kk]<=E->lmesh.nel))
		{
			ival=icheck_element_column(E,j,neighbor[kk],cc,rad);
			if (ival>0)
			{
				return neighbor[kk];
			}
		}
	}
	
    return UNDEFINED_ELEMENT;
}


/********** ICHECK ALL COLUMNS ********************************/
/*                                                            */
/* This function check all columns until the proper one for   */
/* a point (x,y,z) is found. The surface element is returned  */
/* else UNDEFINED_ELEMENT if can't be found.                  */

static int icheck_all_columns(struct All_variables *E,
                              int j,
                              CartesianCoord cc,
                              double rad)
{
    int icheck;
    int nel;
    int elz=E->lmesh.elz;
    int numel=E->lmesh.nel;
	
    for (nel=elz;nel<=numel;nel+=elz)
	{
		icheck=icheck_element_column(E,j,nel,cc,rad);
		if (icheck==1) return nel;
	}
	
    return UNDEFINED_ELEMENT;
}


/******* ICHECK ELEMENT *************************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given element                                          */

static int icheck_element(struct All_variables *E,
                          int j, ElementID nel,
                          CartesianCoord cc,
                          double rad)
{
    int icheck;
	
    icheck = icheck_shell(E, nel, rad);
    if (icheck == 0) return 0;
	
    icheck = icheck_element_column(E, j, nel, cc, rad);
    if (icheck == 0) return 0;
	
    return 1;
}


/********  ICHECK SHELL ************************************/
/*                                                         */
/* This function serves to check whether a point lies      */
/* within the proper radial shell of a given element       */
/* note: j set to 1; shouldn't depend on cap               */

static int icheck_shell(struct All_variables *E,
                        ElementID nel, double rad)
{
    int ibottom_node, itop_node;
    double bottom_rad, top_rad;

    ibottom_node=E->ien[1][nel].node[1];
    itop_node=E->ien[1][nel].node[5];

    bottom_rad=E->sx[1][3][ibottom_node];
    top_rad=E->sx[1][3][itop_node];

    if ((rad>=bottom_rad)&&(rad<top_rad)) return 1;
	else return 0;
}

/********  ICHECK ELEMENT COLUMN ****************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given element's column                                 */

static int icheck_element_column(struct All_variables *E,
                                 int j, ElementID nel,
                                 CartesianCoord cc,
                                 double rad)
{
    CartesianCoord	test_point;
	CapBoundary		bounds;
    int				lev, kk, node;
	
	lev = E->mesh.levmax;
    E->trace.istat_elements_checked++;
	
    /* surface coords of element nodes */
	
    for (kk=0;kk<4;kk++)
	{
		node=E->ien[j][nel].node[kk+4+1];
		
		bounds.setCartTrigBounds(kk,
								 CartesianCoord(E->x[j][1][node], E->x[j][2][node], E->x[j][3][node]),
								 E->SinCos[lev][j][2][node], /* cos(theta) */
								 E->SinCos[lev][j][0][node], /* sin(theta) */
								 E->SinCos[lev][j][3][node], /* cos(phi) */
								 E->SinCos[lev][j][1][node]); /* sin(phi) */
	}
	
    /* test_point - project to outer radius */
	
    test_point = cc/rad;
	
    return icheck_bounds(E,test_point,bounds);
}


/********* ICHECK CAP ***************************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given cap                                              */
/*                                                          */
int full_icheck_cap(struct All_variables *E, int icap,
                    CartesianCoord cc, double rad)
{
	
    CartesianCoord	test_point;
	
    /* test_point - project to outer radius */
	
    test_point = cc/rad;
	
    return icheck_bounds(E,test_point,E->trace.boundaries[icap]);
}

/***** ICHECK BOUNDS ******************************/
/*                                                */
/* This function check if a test_point is bounded */
/* by 4 nodes                                     */
/* This is done by:                               */
/* 1) generate vectors from node to node          */
/* 2) generate vectors from each node to point    */
/*    in question                                 */
/* 3) for each node, take cross product of vector */
/*    pointing to it from previous node and       */
/*    vector from node to point in question       */
/* 4) Find radial components of all the cross     */
/*    products.                                   */
/* 5) If all radial components are positive,      */
/*    point is bounded by the 4 nodes             */
/* 6) If some radial components are negative      */
/*    point is on a boundary - adjust it an       */
/*    epsilon amount for this analysis only       */
/*    which will force it to lie in one element   */
/*    or cap                                      */

static int icheck_bounds(struct All_variables *E,
                         CartesianCoord test_point,
                         const CapBoundary bounds)
{
    int					number_of_tries=0;
	CartesianCoord		v12, v23, v34, v41;
	CartesianCoord		v1p, v2p, v3p, v4p;
	CartesianCoord		cross1, cross2, cross3, cross4;
	SphericalCoord		sc;
	
    double rad1,rad2,rad3,rad4;
    double tiny, eps;
	
    /* make vectors from node to node */
	
    v12 = bounds[1].cartesian() - bounds[0].cartesian();
    v23 = bounds[2].cartesian() - bounds[1].cartesian();
    v34 = bounds[3].cartesian() - bounds[2].cartesian();
    v41 = bounds[0].cartesian() - bounds[3].cartesian();
	
try_again:
	
    number_of_tries++;
	
    /* make vectors from test point to node */
	
	v1p = test_point - bounds[0].cartesian();
	v2p = test_point - bounds[1].cartesian();
	v3p = test_point - bounds[2].cartesian();
	v4p = test_point - bounds[3].cartesian();
	
    /* Calculate cross products */
	
    cross2 = v12.crossProduct(v2p);
    cross3 = v23.crossProduct(v3p);
    cross4 = v34.crossProduct(v4p);
    cross1 = v41.crossProduct(v1p);
	
    /* Calculate radial component of cross products */
	
    rad1=findradial(cross1,bounds[0]);
    rad2=findradial(cross2,bounds[1]);
    rad3=findradial(cross3,bounds[2]);
    rad4=findradial(cross4,bounds[3]);
	
    /*  Check if any radial components is zero (along a boundary), adjust if so */
    /*  Hopefully, this doesn't happen often, may be expensive                  */
	
    tiny=1e-15;
    eps=1e-6;
	
    if (number_of_tries>3)
	{
		fprintf(E->trace.fpt,"Error(icheck_bounds)-too many tries\n");
		fprintf(E->trace.fpt,"Rads: %f %f %f %f\n",rad1,rad2,rad3,rad4);
		fprintf(E->trace.fpt,"Test Point: %f %f %f  \n",test_point._x,test_point._y,test_point._z);
		//fprintf(E->trace.fpt,"Nodal points: 1: %f %f %f\n",rnode1._x,rnode1._y,rnode1._z);
		//fprintf(E->trace.fpt,"Nodal points: 2: %f %f %f\n",rnode2._x,rnode2._y,rnode2._z);
		//fprintf(E->trace.fpt,"Nodal points: 3: %f %f %f\n",rnode3._x,rnode3._y,rnode3._z);
		//fprintf(E->trace.fpt,"Nodal points: 4: %f %f %f\n",rnode4._x,rnode4._y,rnode4._z);
		fflush(E->trace.fpt);
		exit(10);
	}
	
	// If any radial component is small, we are on a boundary
    if (fabs(rad1)<=tiny||fabs(rad2)<=tiny||fabs(rad3)<=tiny||fabs(rad4)<=tiny)
	{
		// Convert our test point to spherical coordinates
		sc = test_point.toSpherical();
		
		// Ensure it is within bounds
		if (sc._theta < 0) sc._theta += 2*M_PI;
		sc._rad = 1;
		
		// Nudge the point by eps
		if (sc._theta <= M_PI/2.0) sc._theta += eps;
		else sc._theta -= eps;
		sc._phi += eps;
		
		// Convert the nudged test point back to Cartesian coordinates
		test_point = sc.toCartesian();
		
		number_of_tries++;
		goto try_again;
	}
	
    if (rad1>0.0 && rad2>0.0 && rad3>0.0 && rad4>0.0) return 1;
	else return 0;
	
    /*
	 fprintf(stderr,"%d: icheck: %d\n",E->parallel.me,icheck);
	 fprintf(stderr,"%d: rads: %f %f %f %f\n",E->parallel.me,rad1,rad2,rad3,rad4);
	 */
}

/****************************************************************************/
/* FINDRADIAL                                                              */
/*                                                                          */
/* This function finds the radial component of a Cartesian vector           */

static double findradial(CartesianCoord vec,
                         BoundaryPoint bp)
{
    double radialparti,radialpartj,radialpartk;

    radialparti=vec._x*bp.sin_theta*bp.cos_phi;
    radialpartj=vec._y*bp.sin_theta*bp.sin_phi;
    radialpartk=vec._z*bp.cos_theta;

	return radialparti+radialpartj+radialpartk;
}


/******************************************************************/
/* FIX ANGLE                                                      */
/*                                                                */
/* This function constrains the value of angle to be              */
/* between 0 and 2 PI                                             */
/*                                                                */

static void fix_angle(double *angle)
{
    const double two_pi = 2.0*M_PI;

    double d2 = floor(*angle / two_pi);

    *angle -= two_pi * d2;
}

/********** IGET ELEMENT *****************************************/
/*                                                               */
/* This function returns the the real element for a given point. */
/* Returns UNDEFINED_ELEMENT if not in this cap.                 */
/* Returns -1 if in this cap but cannot find the element.        */
/* iprevious_element, if known, is the last known element. If    */
/* it is not known, input a negative number.                     */

int full_iget_element(struct All_variables *E,
                      int j, int iprevious_element,
                      CartesianCoord cc,
                      SphericalCoord sc)
{
    int iregel;
    int iel;
    int ntheta,nphi;
    int ival;
    int ichoice;
    int kk;
    int ineighbor;
    int icorner[5];
    int elx,ely,elz,elxz;
    int ifinal_iel;
    int nelem;
	
    elx=E->lmesh.elx;
    ely=E->lmesh.ely;
    elz=E->lmesh.elz;
	
    ntheta=0;
    nphi=0;
	
    /* check the radial range */
    if (E->parallel.nprocz>1)
	{
		ival=icheck_processor_shell(E,j,sc._rad);
		if (ival!=1) return UNDEFINED_ELEMENT;
	}
	
    /* do quick search to see if element can be easily found. */
    /* note that element may still be out of this cap, but    */
    /* it is probably fast to do a quick search before        */
    /* checking cap                                           */
	
    /* get regular element number */
	
    iregel=iget_regel(E,j,sc._theta,sc._phi,&ntheta,&nphi);
    if (iregel<=0)
	{
		return UNDEFINED_ELEMENT;
	}
	
	
    /* AKMA put safety here or in make grid */
	
    if (E->trace.regtoel[j][0][iregel]==0)
	{
		iel=E->trace.regtoel[j][1][iregel];
		goto foundit;
	}
	
    /* first check previous element */
	
    if (iprevious_element>0)
	{
		ival=icheck_element_column(E,j,iprevious_element,cc,sc._rad);
		if (ival==1)
		{
			iel=iprevious_element;
			goto foundit;
		}
	}
	
    /* Check all regular mapping choices */
	
    ichoice=0;
    if (E->trace.regtoel[j][0][iregel]>0)
	{
		
		ichoice=E->trace.regtoel[j][0][iregel];
		for (kk=1;kk<=ichoice;kk++)
		{
			nelem=E->trace.regtoel[j][kk][iregel];
			
			if (nelem!=iprevious_element)
			{
				ival=icheck_element_column(E,j,nelem,cc,sc._rad);
				if (ival==1)
				{
					iel=nelem;
					goto foundit;
				}
				
			}
		}
	}
	
    /* If here, it means that tracer could not be found quickly with regular element map */
	
    /* First check previous element neighbors */
	
    if (iprevious_element>0)
	{
		iel=icheck_column_neighbors(E,j,iprevious_element,cc,sc._rad);
		if (iel>0)
		{
			goto foundit;
		}
	}
	
    /* check if still in cap */
	
    ival=full_icheck_cap(E,0,cc,sc._rad);
    if (ival==0) return UNDEFINED_ELEMENT;
	
    /* if here, still in cap (hopefully, without a doubt) */
	
    /* check cap corners (they are sometimes tricky) */
	
    elxz=elx*elz;
    icorner[1]=elz;
    icorner[2]=elxz;
    icorner[3]=elxz*(ely-1)+elz;
    icorner[4]=elxz*ely;
    for (kk=1;kk<=4;kk++)
	{
		ival=icheck_element_column(E,j,icorner[kk],cc,sc._rad);
		if (ival>0)
		{
			iel=icorner[kk];
			goto foundit;
		}
	}
	
	
    /* if previous element is not known, check neighbors of those tried in iquick... */
	
    if (iprevious_element<0)
	{
		if (ichoice>0)
		{
			for (kk=1;kk<=ichoice;kk++)
			{
				ineighbor=E->trace.regtoel[j][kk][iregel];
				iel=icheck_column_neighbors(E,j,ineighbor,cc,sc._rad);
				if (iel>0)
				{
					goto foundit;
				}
			}
		}
		
	}
	
    /* As a last resort, check all element columns */
	
    E->trace.istat1++;
	
    iel=icheck_all_columns(E,j,cc,sc._rad);
	
    /*
	 fprintf(E->trace.fpt,"WARNING(full_iget_element)-doing a full search!\n");
	 fprintf(E->trace.fpt,"  Most often means tracers have moved more than 1 element away\n");
	 fprintf(E->trace.fpt,"  or regular element resolution is way too low.\n");
	 fprintf(E->trace.fpt,"  COLUMN: %d \n",iel);
	 fprintf(E->trace.fpt,"  PREVIOUS ELEMENT: %d \n",iprevious_element);
	 fprintf(E->trace.fpt,"  x,y,z,theta,phi,rad: %f %f %f   %f %f %f\n",x,y,z,theta,phi,rad);
	 fflush(E->trace.fpt);
	 if (E->trace.itracer_warnings) exit(10);
	 */
	
    if (E->trace.istat1%100==0)
	{
		fprintf(E->trace.fpt,"Checked all elements %d times already this turn\n",E->trace.istat1);
		fflush(E->trace.fpt);
	}
    if (iel>0)
	{
		goto foundit;
	}
	
	
    /* if still here, there is a problem */
	
    fprintf(E->trace.fpt,"Error(full_iget_element) - element not found\n");
    fprintf(E->trace.fpt,"x,y,z,theta,phi,iregel %.15e %.15e %.15e %.15e %.15e %d\n",
            cc._x,cc._y,cc._z,sc._theta,sc._phi,iregel);
    fflush(E->trace.fpt);
    return -1;
	
foundit:
	
    /* find radial element */
	
    ifinal_iel=iget_radial_element(E,j,iel,sc._rad);
	
    return ifinal_iel;
}


/***** IGET RADIAL ELEMENT ***********************************/
/*                                                           */
/* This function returns the proper radial element, given    */
/* an element (iel) from the column.                         */

static int iget_radial_element(struct All_variables *E,
                               int j, int iel,
                               double rad)
{
    int elz=E->lmesh.elz;
    int ibottom_element;
    int iradial_element;
    int node;
    int kk;
    int idum;

    double top_rad;

    /* first project to the lowest element in the column */

    idum=(iel-1)/elz;
    ibottom_element=idum*elz+1;

    iradial_element=ibottom_element;

    for (kk=1;kk<=elz;kk++)
        {

            node=E->ien[j][iradial_element].node[8];
            top_rad=E->sx[j][3][node];

            if (rad<top_rad) goto found_it;

            iradial_element++;

        } /* end kk */


    /* should not be here */

    fprintf(E->trace.fpt,"Error(iget_radial_element)-out of range %f %d %d %d\n",rad,j,iel,ibottom_element);
    fflush(E->trace.fpt);
    exit(10);

 found_it:

    return iradial_element;
}


/*********** IGET REGEL ******************************************/
/*                                                               */
/* This function returns the regular element in which a point    */
/* exists. If not found, returns UNDEFINED_ELEMENT.              */
/* npi and ntheta are modified for later use                     */

static int iget_regel(struct All_variables *E, int j,
                      double theta, double phi,
                      int *ntheta, int *nphi)
{
    int iregel;
    int idum;
    double rdum;

    /* first check whether theta is in range */

    if (theta<E->trace.thetamin[j]) return UNDEFINED_ELEMENT;
    if (theta>E->trace.thetamax[j]) return UNDEFINED_ELEMENT;

    /* get ntheta, nphi on regular mesh */

    rdum=theta-E->trace.thetamin[j];
    idum=rdum/E->trace.deltheta[j];
    *ntheta=idum+1;

    rdum=phi-E->trace.phimin[j];
    idum=rdum/E->trace.delphi[j];
    *nphi=idum+1;

    iregel=*ntheta+(*nphi-1)*E->trace.numtheta[j];

    /* check range to be sure */

    if (iregel>E->trace.numregel[j]) return UNDEFINED_ELEMENT;
    if (iregel<1) return UNDEFINED_ELEMENT;

    return iregel;
}



/****************************************************************/
/* DEFINE UV SPACE                                              */
/*                                                              */
/* This function defines nodal points as orthodrome coordinates */
/* u and v.  In uv space, great circles form straight lines.    */
/* This is used for interpolation method 1                      */
/* E->gnomonic[node].u = u                                      */
/* E->gnomonic[node].v = v                                      */

void define_uv_space(struct All_variables *E)
{
    const int j = 1;
    const int lev = E->mesh.levmax;
    int refnode;
    int i, n;

    double u, v, cosc, theta_f, phi_f, dphi, cosd;
    double *cost, *sint, *cosf, *sinf;

    sint = E->SinCos[lev][j][0];
    sinf = E->SinCos[lev][j][1];
    cost = E->SinCos[lev][j][2];
    cosf = E->SinCos[lev][j][3];

    /* uv space requires a reference point */
    /* use the point at middle of the cap */
    refnode = 1 + E->lmesh.noz * ((E->lmesh.noy / 2) * E->lmesh.nox
                                  + E->lmesh.nox / 2);
    phi_f = E->trace.gnomonic_reference_phi = E->sx[j][2][refnode];

    /** debug **
    theta_f = E->sx[j][1][refnode];
    for (i=1; i<=E->lmesh.nsf; i++) {
        fprintf(E->trace.fpt, "i=%d (%e %e %e %e)\n",
                i, sint[i], sinf[i], cost[i], cosf[i]);
    }
    fprintf(E->trace.fpt, "%d %d %d ref=(%e %e)\n",
            E->lmesh.noz, E->lmesh.nsf, refnode, theta_f, phi_f);
    /**/

    /* store cos(theta_f) and sin(theta_f) */
	E->trace.gnomonic->insert(std::make_pair(0, CoordUV(cost[refnode], sint[refnode])));

    /* convert each nodal point to u and v */

    for (i=1, n=1; i<=E->lmesh.nsf; i++, n+=E->lmesh.noz) {
        dphi = E->sx[j][2][n] - phi_f;
        cosd = cos(dphi);
        cosc = cost[refnode] * cost[n] + sint[refnode] * sint[n] * cosd;
        u = sint[n] * sin(dphi) / cosc;
        v = (sint[refnode] * cost[n] - cost[refnode] * sint[n] * cosd)
            / cosc;

		E->trace.gnomonic->insert(std::make_pair(i, CoordUV(u, v)));

        /** debug **
        fprintf(E->trace.fpt, "n=%d ns=%d cosc=%e (%e %e) -> (%e %e)\n",
                n, i, cosc, E->sx[j][1][n], E->sx[j][2][n], u, v);
        /**/
    }
}

/***************************************************************/
/* DETERMINE SHAPE COEFFICIENTS                                */
/*                                                             */
/* An initialization function that determines the coeffiecients*/
/* to all element shape functions.                             */
/* This method uses standard linear shape functions of         */
/* triangular elements. (See Cuvelier, Segal, and              */
/* van Steenhoven, 1986). This is all in UV space.             */
/*                                                             */
/* shape_coefs[cap][wedge][3 shape functions*3 coefs][nelems]  */

void determine_shape_coefficients(struct All_variables *E)
{
    const int j = 1;
    int nelem, iwedge, kk, i;
    int snode;

	CoordUV	uv[5], xy1, xy2, xy3;

    /* first, allocate memory */

    for(iwedge=0; iwedge<2; iwedge++) {
		if ((E->trace.shape_coefs[j][iwedge] = 
			 (TriElemLinearShapeFunc **)malloc((E->lmesh.snel+1)*sizeof(TriElemLinearShapeFunc*))) == NULL) {
			fprintf(E->trace.fpt,"ERROR(find shape coefs)-not enough memory(a)\n");
			fflush(E->trace.fpt);
			exit(10);
		}
    }

    for (i=1, nelem=1; i<=E->lmesh.snel; i++, nelem+=E->lmesh.elz) {

        /* find u,v of local nodes at one radius  */

        for(kk=1; kk<=4; kk++) {
            snode = (E->ien[j][nelem].node[kk]-1) / E->lmesh.noz + 1;
            uv[kk] = (*E->trace.gnomonic)[snode];
        }

        for(iwedge=0; iwedge<2; iwedge++) {

            if (iwedge == 0) {
                xy1 = uv[1];
                xy2 = uv[2];
                xy3 = uv[3];
            } else if (iwedge == 1) {
                xy1 = uv[1];
                xy2 = uv[3];
                xy3 = uv[4];
            }
			
			E->trace.shape_coefs[j][iwedge][i] = new TriElemLinearShapeFunc(xy1, xy2, xy3);

            /** debug **
            fprintf(E->trace.fpt, "el=%d els=%d iwedge=%d shape=(%e %e %e, %e %e %e, %e %e %e)\n",
                    nelem, i, iwedge,
                    E->trace.shape_coefs[j][iwedge][0][i],
                    E->trace.shape_coefs[j][iwedge][1][i],
                    E->trace.shape_coefs[j][iwedge][2][i],
                    E->trace.shape_coefs[j][iwedge][3][i],
                    E->trace.shape_coefs[j][iwedge][4][i],
                    E->trace.shape_coefs[j][iwedge][5][i],
                    E->trace.shape_coefs[j][iwedge][6][i],
                    E->trace.shape_coefs[j][iwedge][7][i],
                    E->trace.shape_coefs[j][iwedge][8][i]);
            /**/

        } /* end wedge */
    } /* end elem */
}


/* &&&&&&&&&&&&&&&&&&&& ANALYTICAL TESTS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************/

/**************** ANALYTICAL TEST *********************************************************/
/*                                                                                        */
/* This function (and the 2 following) are used to test advection of tracers by assigning */
/* a test function (in "analytical_test_function").                                       */

void analytical_test(struct All_variables *E)
{
#if 0
    int kk;
    int nsteps;
    int j;
    int my_number,number;
    int nrunge_steps;
    int nrunge_refinement;

    double dt;
    double runge_dt;
    double theta,phi,rad;
    double time;
    double my_theta0,my_phi0,my_rad0;
    double my_thetaf,my_phif,my_radf;
    double theta0,phi0,rad0;
    double thetaf,phif,radf;
	SphericalCoord	vel_s, x0_s, xf_s;
	CartesianCoord	vel_c, x0_c, xf_c;
    double vec[4];
    double runge_path_length,runge_time;
    double x0,y0,z0;
    double difference;
    double difperpath;
	TracerList::iterator		tr;

    fprintf(E->trace.fpt,"Starting Analytical Test\n");
    if (E->parallel.me==0) fprintf(stderr,"Starting Analytical Test\n");
    fflush(E->trace.fpt);

    /* Reset Box cushion to 0 */

    E->trace.box_cushion=0.0000;

    /* test paramters */

    nsteps=200;
    dt=0.0001;

    E->advection.timestep=dt;

    fprintf(E->trace.fpt,"steps: %d  dt: %f\n",nsteps,dt);

    /* Assign test velocity to Citcom nodes */

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            for (kk=1;kk<=E->lmesh.nno;kk++)
                {

                    theta=E->sx[j][1][kk];
                    phi=E->sx[j][2][kk];
                    rad=E->sx[j][3][kk];

                    analytical_test_function(E,SphericalCoord(theta,phi,rad),vel_s,vel_c);

                    E->sphere.cap[j].V[1][kk]=vel_s._theta;
                    E->sphere.cap[j].V[2][kk]=vel_s._phi;
                    E->sphere.cap[j].V[3][kk]=vel_s._rad;
                }
        }

    time=0.0;

    my_theta0=0.0;
    my_phi0=0.0;
    my_rad0=0.0;
    my_thetaf=0.0;
    my_phif=0.0;
    my_radf=0.0;

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            if (E->trace.tracers[j].size()>10)
                {
                    fprintf(E->trace.fpt,"Warning(analytical)-too many tracers to print!\n");
                    fflush(E->trace.fpt);
                    if (E->trace.itracer_warnings) exit(10);
                }
        }

    /* print initial positions */

    E->monitor.solution_cycles=0;
    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
			bool first = true;
			for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();++tr)
                {
                    theta=tr->theta();
                    phi=tr->phi();
                    rad=tr->rad();

                    fprintf(E->trace.fpt,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                    if (first) fprintf(stderr,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                    if (first)
                        {
                            my_theta0=theta;
                            my_phi0=phi;
                            my_rad0=rad;
                        }
					first = false;
                }
        }

    /* advect tracers */

    for (kk=1;kk<=nsteps;kk++)
        {
            E->monitor.solution_cycles=kk;

            time += dt;

            predict_tracers(E);
            correct_tracers(E);

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
					bool first = true;
					for (tr=E->trace.tracers[j].begin();tr!=E->trace.tracers[j].end();++tr)
                        {
                            theta=tr->theta();
                            phi=tr->phi();
                            rad=tr->rad();

                            fprintf(E->trace.fpt,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                            if (first) fprintf(stderr,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                            if ((kk==nsteps) && (first))
                                {
                                    my_thetaf=theta;
                                    my_phif=phi;
                                    my_radf=rad;
                                }
                        }
                }

        }

    /* Get ready for comparison to Runge-Kutte (only works for one tracer) */

    fflush(E->trace.fpt);
    parallel_process_sync(E);

    fprintf(E->trace.fpt,"\n\nComparison to Runge-Kutte\n");
    if (E->parallel.me==0) fprintf(stderr,"Comparison to Runge-Kutte\n");

    for (j=1;j<=E->sphere.caps_per_proc;j++)
        {
            my_number=E->trace.tracers[j].size();
        }

    MPI_Allreduce(&my_number,&number,1,MPI_INT,MPI_SUM,E->parallel.world);

    fprintf(E->trace.fpt,"Number of tracers: %d\n", number);
    if (E->parallel.me==0) fprintf(stderr,"Number of tracers: %d\n", number);

    /* if more than 1 tracer, exit */

    if (number!=1)
        {
            fprintf(E->trace.fpt,"(Note: RK comparison only appropriate for one tracing particle (%d here) \n",number);
            if (E->parallel.me==0) fprintf(stderr,"(Note: RK comparison only appropriate for one tracing particle (%d here) \n",number);
            fflush(E->trace.fpt);
            parallel_process_termination();
        }

    /* communicate starting and final positions */

    MPI_Allreduce(&my_theta0,&theta0,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_phi0,&phi0,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_rad0,&rad0,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_thetaf,&thetaf,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_phif,&phif,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);
    MPI_Allreduce(&my_radf,&radf,1,MPI_DOUBLE,MPI_SUM,E->parallel.world);

	x0_s = SphericalCoord(theta0, phi0, rad0);

    nrunge_refinement=1000;

    nrunge_steps=nsteps*nrunge_refinement;
    runge_dt=dt/(1.0*nrunge_refinement);

    analytical_runge_kutte(E,nrunge_steps,runge_dt,x0_s,x0_c,xf_s,xf_c,vec);

    runge_time=vec[1];
    runge_path_length=vec[2];

    /* initial coordinates - both citcom and RK */

	CartesianCoord	ccf;
	SphericalCoord	scf(thetaf, phif, radf);
	
    x0=x0_c._x;
    y0=x0_c._y;
    z0=x0_c._z;

    /* convert final citcom coords into cartesian */

	ccf = scf.toCartesian();

    difference=ccf.dist(xf_c);

    difperpath=difference/runge_path_length;

    /* Print out results */

    fprintf(E->trace.fpt,"Citcom calculation: steps: %d  dt: %f\n",nsteps,dt);
    fprintf(E->trace.fpt,"  (nodes per cap: %d x %d x %d)\n",E->lmesh.nox,E->lmesh.noy,(E->lmesh.noz-1)*E->parallel.nprocz+1);
    fprintf(E->trace.fpt,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
    fprintf(E->trace.fpt,"                    final position: theta: %f phi: %f rad: %f\n", thetaf,phif,radf);
    fprintf(E->trace.fpt,"                    (final time: %f) \n",time );

    fprintf(E->trace.fpt,"\n\nRunge-Kutte calculation: steps: %d  dt: %g\n",nrunge_steps,runge_dt);
    fprintf(E->trace.fpt,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
    fprintf(E->trace.fpt,"                    final position: theta: %f phi: %f rad: %f\n",xf_s._theta,xf_s._phi,xf_s._rad);
    fprintf(E->trace.fpt,"                    path length: %f \n",runge_path_length );
    fprintf(E->trace.fpt,"                    (final time: %f) \n",runge_time );

    fprintf(E->trace.fpt,"\n\n Difference between Citcom and RK: %e  (diff per path length: %e)\n\n",difference,difperpath);

    if (E->parallel.me==0)
        {
            fprintf(stderr,"Citcom calculation: steps: %d  dt: %f\n",nsteps,dt);
            fprintf(stderr,"  (nodes per cap: %d x %d x %d)\n",E->lmesh.nox,E->lmesh.noy,(E->lmesh.noz-1)*E->parallel.nprocz+1);
            fprintf(stderr,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
            fprintf(stderr,"                    final position: theta: %f phi: %f rad: %f\n", thetaf,phif,radf);
            fprintf(stderr,"                    (final time: %f) \n",time );

            fprintf(stderr,"\n\nRunge-Kutte calculation: steps: %d  dt: %f\n",nrunge_steps,runge_dt);
            fprintf(stderr,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
            fprintf(stderr,"                    final position: theta: %f phi: %f rad: %f\n",xf_s._theta,xf_s._phi,xf_s._rad);
            fprintf(stderr,"                    path length: %f \n",runge_path_length );
            fprintf(stderr,"                    (final time: %f) \n",runge_time );

            fprintf(stderr,"\n\n Difference between Citcom and RK: %e  (diff per path length: %e)\n\n",difference,difperpath);

        }

    fflush(E->trace.fpt);
#endif
}

/*************** ANALYTICAL RUNGE KUTTE ******************/
/*                                                       */
void analytical_runge_kutte(struct All_variables *E, int nsteps, double dt,
							SphericalCoord &x0_s, CartesianCoord &x0_c,
							SphericalCoord &xf_s, CartesianCoord &xf_c, double *vec)
{
    int					kk;
	CartesianCoord		cc0, ccp, ccc, vel0_c, velp_c;
	SphericalCoord		sc0, scp, scc, vel0_s, velp_s;
    double				time, path;
	
	sc0 = x0_s;
	cc0 = sc0.toCartesian();
	
    /* fill initial cartesian vector to send back */
	x0_c = cc0;
	
    time=0.0;
    path=0.0;
	
    for (kk=0;kk<nsteps;kk++)
	{
		// Get velocity at initial position
		analytical_test_function(E,sc0,vel0_s,vel0_c);
		
		// Find predicted midpoint position
		ccp = cc0 + vel0_c * (dt*0.5);
		
		// Convert to spherical
		scp = ccp.toSpherical();
		
		// Get velocity at predicted midpoint position
		analytical_test_function(E,scp,velp_s,velp_c);
		
		// Find corrected position using midpoint velocity
		ccc = cc0 + velp_c * dt;
		
		// Convert to spherical
		scc = ccc.toSpherical();
		
		// Compute path length
		path += ccc.dist(cc0);
		
		time += dt;
		
		cc0 = ccc;
		
		/* next time step */
	}
	
    /* fill final spherical and cartesian vectors to send back */
	
	xf_s = scc;
	xf_c = ccc;
	
    vec[1]=time;
    vec[2]=path;
}



/**************** ANALYTICAL TEST FUNCTION ******************/
/*                                                          */
/* vel_s => velocity in spherical directions                */
/*                                                          */
/* vel_c => velocity in cartesian directions                */

void analytical_test_function(struct All_variables *E, SphericalCoord sc, SphericalCoord &vel_s, CartesianCoord &vel_c)
{
    /* This is where the function is given in spherical */
    vel_s._theta=50.0*sc._rad*cos(sc._phi);
    vel_s._phi=100.0*sc._rad*sin(sc._theta);
    vel_s._rad=25.0*sc._rad;

    /* Convert the function into cartesian */
	vel_c = vel_s.toCartesian();
}


/**** PDEBUG ***********************************************************/
void pdebug(struct All_variables *E, int i)
{

    fprintf(E->trace.fpt,"HERE (Before Sync): %d\n",i);
    fflush(E->trace.fpt);
    parallel_process_sync(E);
    fprintf(E->trace.fpt,"HERE (After Sync): %d\n",i);
    fflush(E->trace.fpt);

    return;
}
