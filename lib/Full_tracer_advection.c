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
                        int j, int nelem,
                        double u, double v,
                        int iwedge, double * shape2d);
static void get_radial_shape(struct All_variables *E,
                             int j, int nelem,
                             double rad, double *shaperad);
static void spherical_to_uv(struct All_variables *E, int j,
                            double theta, double phi,
                            double *u, double *v);
static void make_regular_grid(struct All_variables *E);
static void write_trace_instructions(struct All_variables *E);
static int icheck_column_neighbors(struct All_variables *E,
                                   int j, int nel,
                                   double x, double y, double z,
                                   double rad);
static int icheck_all_columns(struct All_variables *E,
                              int j,
                              double x, double y, double z,
                              double rad);
static int icheck_element(struct All_variables *E,
                          int j, int nel,
                          double x, double y, double z,
                          double rad);
static int icheck_shell(struct All_variables *E,
                        int nel, double rad);
static int icheck_element_column(struct All_variables *E,
                                 int j, int nel,
                                 double x, double y, double z,
                                 double rad);
static int icheck_bounds(struct All_variables *E,
                         double *test_point,
                         double *rnode1, double *rnode2,
                         double *rnode3, double *rnode4);
static double findradial(struct All_variables *E, double *vec,
                         double cost, double sint,
                         double cosf, double sinf);
static void makevector(double *vec, double xf, double yf, double zf,
                       double x0, double y0, double z0);
static void crossit(double *cross, double *A, double *B);
static void fix_radius(struct All_variables *E,
                       double *radius, double *theta, double *phi,
                       double *x, double *y, double *z);
static void fix_phi(double *phi);
static void fix_theta_phi(double *theta, double *phi);
static int iget_radial_element(struct All_variables *E,
                               int j, int iel,
                               double rad);
static int iget_regel(struct All_variables *E, int j,
                      double theta, double phi,
                      int *ntheta, int *nphi);
static void define_uv_space(struct All_variables *E);
static void determine_shape_coefficients(struct All_variables *E);
static void full_put_lost_tracers(struct All_variables *E,
                                  int isend[13][13], double *send[13][13]);
void pdebug(struct All_variables *E, int i);
int full_icheck_cap(struct All_variables *E, int icap,
                    double x, double y, double z, double rad);



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


    return;
}

/***** FULL TRACER SETUP ************************/

void full_tracer_setup(struct All_variables *E)
{

    char output_file[200];
    void get_neighboring_caps();
    void analytical_test();
    double CPU_time0();
    double begin_time = CPU_time0();

    /* Some error control */

    if (E->sphere.caps_per_proc>1) {
            fprintf(stderr,"This code does not work for multiple caps per processor!\n");
            parallel_process_termination();
    }


    /* open tracing output file */

    sprintf(output_file,"%s.tracer_log.%d",E->control.data_file,E->parallel.me);
    E->trace.fpt=fopen(output_file,"w");


    /* reset statistical counters */

    E->trace.istat_isend=0;
    E->trace.istat_iempty=0;
    E->trace.istat_elements_checked=0;
    E->trace.istat1=0;


    /* some obscure initial parameters */
    /* This parameter specifies how close a tracer can get to the boundary */
    E->trace.box_cushion=0.00001;

    /* Determine number of tracer quantities */

    /* advection_quantites - those needed for advection */
    E->trace.number_of_basic_quantities=12;

    /* extra_quantities - used for flavors, composition, etc.    */
    /* (can be increased for additional science i.e. tracing chemistry */

    E->trace.number_of_extra_quantities = 0;
    if (E->trace.nflavors > 0)
        E->trace.number_of_extra_quantities += 1;


    E->trace.number_of_tracer_quantities =
        E->trace.number_of_basic_quantities +
        E->trace.number_of_extra_quantities;


    /* Fixed positions in tracer array */
    /* Flavor is always in extraq position 0  */
    /* Current coordinates are always kept in basicq positions 0-5 */
    /* Other positions may be used depending on science being done */


    /* Some error control regarding size of pointer arrays */

    if (E->trace.number_of_basic_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(initialize_trace)-increase 2nd position size of basic in tracer_defs.h\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }
    if (E->trace.number_of_extra_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(initialize_trace)-increase 2nd position size of extraq in tracer_defs.h\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }
    if (E->trace.number_of_tracer_quantities>99) {
        fprintf(E->trace.fpt,"ERROR(initialize_trace)-increase 2nd position size of rlater in tracer_defs.h\n");
        fflush(E->trace.fpt);
        parallel_process_termination();
    }

    write_trace_instructions(E);


    /* Gnometric projection for velocity interpolation */
    define_uv_space(E);
    determine_shape_coefficients(E);


    /* The bounding box of neiboring processors */
    get_neighboring_caps(E);


    /* Fine-grained regular grid to search tracers */
    make_regular_grid(E);


    if (E->trace.ianalytical_tracer_test==1) {
        //TODO: walk into this code...
        analytical_test(E);
        parallel_process_termination();
    }

    if (E->composition.on)
        composition_setup(E);

    fprintf(E->trace.fpt, "Tracer intiailization takes %f seconds.\n",
            CPU_time0() - begin_time);

    return;
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

    void expand_tracer_arrays();
    int icheck_that_processor_shell();

    double CPU_time0();
    double begin_time = CPU_time0();

    int number_of_caps=12;
    int lev=E->mesh.levmax;
    int num_ngb = E->parallel.TNUM_PASS[lev][j];

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


    E->trace.istat_isend=E->trace.ilater[j];
    /* debug */
    if(E->control.verbose){
      for (kk=1; kk<=E->trace.istat_isend; kk++) {
        fprintf(E->trace.fpt, "tracer#=%d xx=(%g,%g,%g)\n", kk,
                E->trace.rlater[j][0][kk],
                E->trace.rlater[j][1][kk],
                E->trace.rlater[j][2][kk]);
      }
      fflush(E->trace.fpt);
    }
    /**/



    /* initialize isend and ireceive */
    /* # of neighbors in the horizontal plane */
    isize[j]=E->trace.ilater[j]*E->trace.number_of_tracer_quantities;
    for (kk=0;kk<=num_ngb;kk++) isend[j][kk]=0;
    for (kk=0;kk<=num_ngb;kk++) ireceive[j][kk]=0;

    /* Allocate Maximum Memory to Send Arrays */

    itemp_size=max(isize[j],1);

    for (kk=0;kk<=num_ngb;kk++) {
        if ((send[j][kk]=(double *)malloc(itemp_size*sizeof(double)))==NULL) {
            fprintf(E->trace.fpt,"Error(lost souls)-no memory (u389)\n");
            fflush(E->trace.fpt);
            exit(10);
        }
    }


    /* debug *
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


    /** debug **/
    if(E->control.verbose){
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
    }
    /**/

    /* Allocate memory in receive arrays */

    for (ithatcap=0;ithatcap<=num_ngb;ithatcap++) {
        isize[j]=ireceive[j][ithatcap]*E->trace.number_of_tracer_quantities;

        itemp_size=max(1,isize[j]);

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
        isize[j]=isend[j][ithatcap]*E->trace.number_of_tracer_quantities;
        for (mm=0;mm<isize[j];mm++) {
            receive[j][ithatcap][mm]=send[j][ithatcap][mm];
        }

    }

    /* neighbor caps */

    for (kk=1;kk<=num_ngb;kk++) {
        idestination_proc=E->parallel.PROCESSOR[lev][j].pass[kk];

        isize[j]=isend[j][kk]*E->trace.number_of_tracer_quantities;

        MPI_Isend(send[j][kk],isize[j],MPI_DOUBLE,idestination_proc,
                  11,E->parallel.world,&request[idb++]);

        isize[j]=ireceive[j][kk]*E->trace.number_of_tracer_quantities;

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

    isize[j]=isum[j]*E->trace.number_of_tracer_quantities;
    isize[j]=max(isize[j],1);
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
            ipos=pp*E->trace.number_of_tracer_quantities;

            for (mm=0;mm<E->trace.number_of_tracer_quantities;mm++) {
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
            isize[j]=itracers_subject_to_vertical_transport[j]*E->trace.number_of_tracer_quantities;
            isize[j]=max(isize[j],1);

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

                ireceive_position=it*E->trace.number_of_tracer_quantities;
                it++;

                irad=ireceive_position+2;

                rad=REC[j][irad];

                ival=icheck_that_processor_shell(E,j,ithat_processor,rad);


                /* if tracer is in other shell, take out of receive array and give to send_z*/

                if (ival==1) {


                    isend_position=isend_z[j][ivertical_neighbor]*E->trace.number_of_tracer_quantities;
                    isend_z[j][ivertical_neighbor]++;

                    ilast_receiver_position=(irec[j]-1)*E->trace.number_of_tracer_quantities;

                    for (mm=0;mm<=(E->trace.number_of_tracer_quantities-1);mm++) {
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
            isize[j]=ireceive_z[j][kk]*E->trace.number_of_tracer_quantities;
            isize[j]=max(isize[j],1);

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

            isize_send=isend_z[j][kk]*E->trace.number_of_tracer_quantities;

            MPI_Isend(send_z[j][kk],isize_send,MPI_DOUBLE,idestination_proc,
                      15,E->parallel.world,&request[idb++]);

            isize_receive=ireceive_z[j][kk]*E->trace.number_of_tracer_quantities;

            MPI_Irecv(receive_z[j][kk],isize_receive,MPI_DOUBLE,idestination_proc,
                      15,E->parallel.world,&request[idb++]);
        }

        /* Wait for non-blocking calls to complete */

        MPI_Waitall(idb,request,status);


        /* Put tracers into REC array */

        /* First, reallocate memory to REC */

        isum[j]=0;
        for (ivertical_neighbor=1;ivertical_neighbor<=E->parallel.TNUM_PASSz[lev];ivertical_neighbor++) {
            isum[j]=isum[j]+ireceive_z[j][ivertical_neighbor];
        }

        isum[j]=isum[j]+irec[j];

        isize[j]=isum[j]*E->trace.number_of_tracer_quantities;

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

                irec_position=irec[j]*E->trace.number_of_tracer_quantities;
                irec[j]++;
                ireceive_position=kk*E->trace.number_of_tracer_quantities;

                for (mm=0;mm<E->trace.number_of_tracer_quantities;mm++) {
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
        E->trace.ntracers[j]++;

        if (E->trace.ntracers[j]>(E->trace.max_ntracers[j]-5)) expand_tracer_arrays(E,j);

        ireceive_position=kk*E->trace.number_of_tracer_quantities;

        for (mm=0;mm<E->trace.number_of_basic_quantities;mm++) {
            ipos=ireceive_position+mm;

            E->trace.basicq[j][mm][E->trace.ntracers[j]]=REC[j][ipos];
        }
        for (mm=0;mm<E->trace.number_of_extra_quantities;mm++) {
            ipos=ireceive_position+E->trace.number_of_basic_quantities+mm;

            E->trace.extraq[j][mm][E->trace.ntracers[j]]=REC[j][ipos];
        }

        theta=E->trace.basicq[j][0][E->trace.ntracers[j]];
        phi=E->trace.basicq[j][1][E->trace.ntracers[j]];
        rad=E->trace.basicq[j][2][E->trace.ntracers[j]];
        x=E->trace.basicq[j][3][E->trace.ntracers[j]];
        y=E->trace.basicq[j][4][E->trace.ntracers[j]];
        z=E->trace.basicq[j][5][E->trace.ntracers[j]];


        iel=(E->trace.iget_element)(E,j,-99,x,y,z,theta,phi,rad);

        if (iel<1) {
            fprintf(E->trace.fpt,"Error(lost souls) - element not here?\n");
            fprintf(E->trace.fpt,"x,y,z-theta,phi,rad: %f %f %f - %f %f %f\n",x,y,z,theta,phi,rad);
            fflush(E->trace.fpt);
            exit(10);
        }

        E->trace.ielement[j][E->trace.ntracers[j]]=iel;

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
    return;
}


static void full_put_lost_tracers(struct All_variables *E,
                                  int isend[13][13], double *send[13][13])
{
    const int j = 1;
    int kk, pp;
    int numtracers, ithatcap, icheck;
    int isend_position, ipos;
    int lev = E->mesh.levmax;
    double theta, phi, rad;
    double x, y, z;

    /* transfer tracers from rlater to send */

    numtracers=E->trace.ilater[j];

    for (kk=1;kk<=numtracers;kk++) {
        rad=E->trace.rlater[j][2][kk];
        x=E->trace.rlater[j][3][kk];
        y=E->trace.rlater[j][4][kk];
        z=E->trace.rlater[j][5][kk];

        /* first check same cap if nprocz>1 */

        if (E->parallel.nprocz>1) {
            ithatcap=0;
            icheck=full_icheck_cap(E,ithatcap,x,y,z,rad);
            if (icheck==1) goto foundit;

        }

        /* check neighboring caps */

        for (pp=1;pp<=E->parallel.TNUM_PASS[lev][j];pp++) {
            ithatcap=pp;
            icheck=full_icheck_cap(E,ithatcap,x,y,z,rad);
            if (icheck==1) goto foundit;
        }


        /* should not be here */
        if (icheck!=1) {
            fprintf(E->trace.fpt,"Error(lost souls)-should not be here\n");
            fprintf(E->trace.fpt,"x: %f y: %f z: %f rad: %f\n",x,y,z,rad);
            icheck=full_icheck_cap(E,0,x,y,z,rad);
            if (icheck==1) fprintf(E->trace.fpt," icheck here!\n");
            else fprintf(E->trace.fpt,"icheck not here!\n");
            fflush(E->trace.fpt);
            exit(10);
        }

    foundit:

        isend[j][ithatcap]++;

        /* assign tracer to send */

        isend_position=(isend[j][ithatcap]-1)*E->trace.number_of_tracer_quantities;

        for (pp=0;pp<=(E->trace.number_of_tracer_quantities-1);pp++) {
            ipos=isend_position+pp;
            send[j][ithatcap][ipos]=E->trace.rlater[j][pp][kk];
        }

    } /* end kk, assigning tracers */

    return;
}


/************************ GET VELOCITY ***************************************/
/*                                                                           */
/* This function interpolates tracer velocity using gnominic interpolation.  */
/* Real theta,phi,rad space is transformed into u,v space. This transformation */
/* maps great circles into straight lines. Here, elements boundaries are     */
/* assumed to be great circle planes (not entirely true, it is actually only */
/* the nodal arrangement that lies upon great circles).  Element boundaries  */
/* are then mapped into planes.  The element is then divided into 2 wedges   */
/* in which standard shape functions are used to interpolate velocity.       */
/* This transformation was found on the internet (refs were difficult to     */
/* to obtain). It was tested that nodal configuration is indeed transformed  */
/* into straight lines.                                                      */
/* The transformations require a reference point along each cap. Here, the   */
/* first point is used.                                                      */
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


void full_get_velocity(struct All_variables *E,
                       int j, int nelem,
                       double theta, double phi, double rad,
                       double *velocity_vector)
{
    int iwedge,inum;
    int i, kk;
    int ival;
    int itry;
    int sphere_key = 0;

    double u,v;
    double shape2d[4];
    double shaperad[3];
    double shape[7];
    double VV[4][9];
    double vx[7],vy[7],vz[7];
    double x,y,z;


    int maxlevel=E->mesh.levmax;

    const double eps=-1e-4;

    void sphere_to_cart();
    void velo_from_element_d();


    /* find u and v using spherical coordinates */

    spherical_to_uv(E,j,theta,phi,&u,&v);

    inum=0;
    itry=1;

 try_again:

    /* Check first wedge (1 of 2) */

    iwedge=1;

 next_wedge:

    /* determine shape functions of wedge */
    /* There are 3 shape functions for the triangular wedge */

    get_2dshape(E,j,nelem,u,v,iwedge,shape2d);

    /* if any shape functions are negative, goto next wedge */

    if (shape2d[1]<eps||shape2d[2]<eps||shape2d[3]<eps)
        {
            inum=inum+1;
            /* AKMA clean this up */
            if (inum>3)
                {
                    fprintf(E->trace.fpt,"ERROR(gnomonic_interpolation)-inum>3!\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }
            if (inum>1 && itry==1)
                {
                    fprintf(E->trace.fpt,"ERROR(gnomonic_interpolation)-inum>1\n");
                    fprintf(E->trace.fpt,"shape %f %f %f\n",shape2d[1],shape2d[2],shape2d[3]);
                    fprintf(E->trace.fpt,"u %f v %f element: %d \n",u,v, nelem);
                    fprintf(E->trace.fpt,"Element uv boundaries: \n");
                    for(kk=1;kk<=4;kk++) {
                        i = (E->ien[j][nelem].node[kk] - 1) / E->lmesh.noz + 1;
                        fprintf(E->trace.fpt,"%d: U: %f V:%f\n",kk,E->gnomonic[i].u,E->gnomonic[i].v);
                    }
                    fprintf(E->trace.fpt,"theta: %f phi: %f rad: %f\n",theta,phi,rad);
                    fprintf(E->trace.fpt,"Element theta-phi boundaries: \n");
                    for(kk=1;kk<=4;kk++)
                        fprintf(E->trace.fpt,"%d: Theta: %f Phi:%f\n",kk,E->sx[j][1][E->ien[j][nelem].node[kk]],E->sx[j][2][E->ien[j][nelem].node[kk]]);
                    sphere_to_cart(E,theta,phi,rad,&x,&y,&z);
                    ival=icheck_element(E,j,nelem,x,y,z,rad);
                    fprintf(E->trace.fpt,"ICHECK?: %d\n",ival);
                    ival=(E->trace.iget_element)(E,j,-99,x,y,z,theta,phi,rad);
                    fprintf(E->trace.fpt,"New Element?: %d\n",ival);
                    ival=icheck_column_neighbors(E,j,nelem,x,y,z,rad);
                    fprintf(E->trace.fpt,"New Element (neighs)?: %d\n",ival);
                    nelem=ival;
                    ival=icheck_element(E,j,nelem,x,y,z,rad);
                    fprintf(E->trace.fpt,"ICHECK?: %d\n",ival);
                    itry++;
                    if (ival>0) goto try_again;
                    fprintf(E->trace.fpt,"NO LUCK\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }

            iwedge=2;
            goto next_wedge;
        }

    /* Determine radial shape functions */
    /* There are 2 shape functions radially */

    get_radial_shape(E,j,nelem,rad,shaperad);

    /* There are 6 nodes to the solid wedge.             */
    /* The 6 shape functions assocated with the 6 nodes  */
    /* are products of radial and wedge shape functions. */

    /* Sum of shape functions is 1                       */

    shape[1]=shaperad[1]*shape2d[1];
    shape[2]=shaperad[1]*shape2d[2];
    shape[3]=shaperad[1]*shape2d[3];
    shape[4]=shaperad[2]*shape2d[1];
    shape[5]=shaperad[2]*shape2d[2];
    shape[6]=shaperad[2]*shape2d[3];

    /* get cartesian velocity */
    velo_from_element_d(E, VV, j, nelem, sphere_key);

    /* depending on wedge, set up velocity points */

    if (iwedge==1)
        {
            vx[1]=VV[1][1];
            vx[2]=VV[1][2];
            vx[3]=VV[1][3];
            vx[4]=VV[1][5];
            vx[5]=VV[1][6];
            vx[6]=VV[1][7];
            vy[1]=VV[2][1];
            vy[2]=VV[2][2];
            vy[3]=VV[2][3];
            vy[4]=VV[2][5];
            vy[5]=VV[2][6];
            vy[6]=VV[2][7];
            vz[1]=VV[3][1];
            vz[2]=VV[3][2];
            vz[3]=VV[3][3];
            vz[4]=VV[3][5];
            vz[5]=VV[3][6];
            vz[6]=VV[3][7];
        }
    if (iwedge==2)
        {
            vx[1]=VV[1][1];
            vx[2]=VV[1][3];
            vx[3]=VV[1][4];
            vx[4]=VV[1][5];
            vx[5]=VV[1][7];
            vx[6]=VV[1][8];
            vy[1]=VV[2][1];
            vy[2]=VV[2][3];
            vy[3]=VV[2][4];
            vy[4]=VV[2][5];
            vy[5]=VV[2][7];
            vy[6]=VV[2][8];
            vz[1]=VV[3][1];
            vz[2]=VV[3][3];
            vz[3]=VV[3][4];
            vz[4]=VV[3][5];
            vz[5]=VV[3][7];
            vz[6]=VV[3][8];
        }

    velocity_vector[1]=vx[1]*shape[1]+vx[2]*shape[2]+shape[3]*vx[3]+
        vx[4]*shape[4]+vx[5]*shape[5]+shape[6]*vx[6];
    velocity_vector[2]=vy[1]*shape[1]+vy[2]*shape[2]+shape[3]*vy[3]+
        vy[4]*shape[4]+vy[5]*shape[5]+shape[6]*vy[6];
    velocity_vector[3]=vz[1]*shape[1]+vz[2]*shape[2]+shape[3]*vz[3]+
        vz[4]*shape[4]+vz[5]*shape[5]+shape[6]*vz[6];



    return;
}

/***************************************************************/
/* GET 2DSHAPE                                                 */
/*                                                             */
/* This function determines shape functions at u,v             */
/* This method uses standard linear shape functions of         */
/* triangular elements. (See Cuvelier, Segal, and              */
/* van Steenhoven, 1986).                                      */

static void get_2dshape(struct All_variables *E,
                        int j, int nelem,
                        double u, double v,
                        int iwedge, double * shape2d)
{

    double a0,a1,a2;
    /* convert nelem to surface element number */
    int n = (nelem - 1) / E->lmesh.elz + 1;

    /* shape function 1 */

    a0=E->trace.shape_coefs[j][iwedge][1][n];
    a1=E->trace.shape_coefs[j][iwedge][2][n];
    a2=E->trace.shape_coefs[j][iwedge][3][n];

    shape2d[1]=a0+a1*u+a2*v;

    /* shape function 2 */

    a0=E->trace.shape_coefs[j][iwedge][4][n];
    a1=E->trace.shape_coefs[j][iwedge][5][n];
    a2=E->trace.shape_coefs[j][iwedge][6][n];

    shape2d[2]=a0+a1*u+a2*v;

    /* shape function 3 */

    a0=E->trace.shape_coefs[j][iwedge][7][n];
    a1=E->trace.shape_coefs[j][iwedge][8][n];
    a2=E->trace.shape_coefs[j][iwedge][9][n];

    shape2d[3]=a0+a1*u+a2*v;

    return;
}

/***************************************************************/
/* GET RADIAL SHAPE                                            */
/*                                                             */
/* This function determines radial shape functions at rad      */

static void get_radial_shape(struct All_variables *E,
                             int j, int nelem,
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
      shaperad[1]=1.0-f1=1.0-(1.0-f2)=f2;
      shaperad[2]=1.0-f2=1.0-(10-f1)=f1;
    */

    shaperad[1]=f2;
    shaperad[2]=f1;

    /* Some error control */

    if (shaperad[1]>(top_bound)||shaperad[1]<(bottom_bound)||
        shaperad[2]>(top_bound)||shaperad[2]<(bottom_bound))
        {
            fprintf(E->trace.fpt,"ERROR(get_radial_shape)\n");
            fprintf(E->trace.fpt,"shaperad[1]: %f \n",shaperad[1]);
            fprintf(E->trace.fpt,"shaperad[2]: %f \n",shaperad[2]);
            fflush(E->trace.fpt);
            exit(10);
        }

    return;
}





/**************************************************************/
/* SPHERICAL TO UV                                               */
/*                                                            */
/* This function transforms theta and phi to new coords       */
/* u and v using gnomonic projection.                          */

static void spherical_to_uv(struct All_variables *E, int j,
                            double theta, double phi,
                            double *u, double *v)
{

    double theta_f;
    double phi_f;
    double cosc;
    double cos_theta_f,sin_theta_f;
    double cost,sint,cosp2,sinp2;

    /* theta_f and phi_f are the reference points at the 1st node of the cap */

    phi_f = E->sx[j][2][1];

    cos_theta_f = E->gnomonic[0].u;
    sin_theta_f = E->gnomonic[0].v;

    cost=cos(theta);
    /*
      sint=sin(theta);
    */
    sint=sqrt(1.0-cost*cost);

    cosp2=cos(phi-phi_f);
    sinp2=sin(phi-phi_f);

    cosc=cos_theta_f*cost+sin_theta_f*sint*cosp2;
    cosc=1.0/cosc;

    *u=sint*sinp2*cosc;
    *v=(sin_theta_f*cost-cos_theta_f*sint*cosp2)*cosc;

    return;
}


/*********** MAKE REGULAR GRID ********************************/
/*                                                            */
/* This function generates the finer regular grid which is    */
/* mapped to real elements                                    */

static void make_regular_grid(struct All_variables *E)
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

    void sphere_to_cart();

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

                    thetamax=max(thetamax,theta);
                    thetamin=min(thetamin,theta);

                }

            /* expand range slightly (should take care of poles)  */

            thetamax=thetamax+expansion;
            thetamax=min(thetamax,M_PI);

            thetamin=thetamin-expansion;
            thetamin=max(thetamin,0.0);

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
            /* the real element value if one exists (-99 otherwise)     */



            if ((E->trace.regnodetoel[j]=(int *)malloc((numregnodes+1)*sizeof(int)))==NULL)
                {
                    fprintf(E->trace.fpt,"ERROR(make regular) -no memory - uh3ud\n");
                    fflush(E->trace.fpt);
                    exit(10);
                }


            /* Initialize regnodetoel - reg elements not used =-99 */

            for (kk=1;kk<=numregnodes;kk++)
                {
                    E->trace.regnodetoel[j][kk]=-99;
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

                            theta_min=min(theta_min,theta);
                            theta_max=max(theta_max,theta);
                            phi_min=min(phi_min,phi);
                            phi_max=max(phi_max,phi);
                        }

                    /* add half difference to phi and expansion to theta to be safe */

                    theta_max=theta_max+expansion;
                    theta_min=theta_min-expansion;

                    theta_max=min(M_PI,theta_max);
                    theta_min=max(0.0,theta_min);

                    half_diff=0.5*(phi_max-phi_min);
                    phi_max=phi_max+half_diff;
                    phi_min=phi_min-half_diff;

                    fix_phi(&phi_max);
                    fix_phi(&phi_min);

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

                    E->trace.regnodetoel[j][kk]=-99;

                    /* find theta and phi for a given regular node */

                    idum1=(kk-1)/(numtheta+1);
                    idum2=kk-1-idum1*(numtheta+1);

                    theta=thetamin+(1.0*idum2*deltheta);
                    phi=phimin+(1.0*idum1*delphi);

                    sphere_to_cart(E,theta,phi,rad,&x,&y,&z);


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

                    ival=icheck_element_column(E,j,ilast_el,x,y,z,rad);
                    if (ival>0)
                        {
                            E->trace.regnodetoel[j][kk]=ilast_el;
                            goto foundit;
                        }

                    /* check neighbors */

                    ival=icheck_column_neighbors(E,j,ilast_el,x,y,z,rad);
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
                                    ival=icheck_element_column(E,j,mm,x,y,z,rad);
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

                    if (E->trace.regnodetoel[j][kk]!=-99)
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
            /*    ichoice=-1   all regular element nodes = -99 (no elements) */
            /*    ichoice=0    all 4 corners within one element              */
            /*    ichoice=1     one element choice (diff from ichoice 0 in    */
            /*                  that perhaps one reg node is in an element    */
            /*                  and the rest are not (-99).                   */
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


    return;
}


/**** WRITE TRACE INSTRUCTIONS ***************/
static void write_trace_instructions(struct All_variables *E)
{
    int i;

    fprintf(E->trace.fpt,"\nTracing Activated! (proc: %d)\n",E->parallel.me);
    fprintf(E->trace.fpt,"   Allen K. McNamara 12-2003\n\n");

    if (E->trace.ic_method==0)
        {
            fprintf(E->trace.fpt,"Generating New Tracer Array\n");
            fprintf(E->trace.fpt,"Tracers per element: %d\n",E->trace.itperel);
        }
    if (E->trace.ic_method==1)
        {
            fprintf(E->trace.fpt,"Reading tracer file %s\n",E->trace.tracer_file);
        }
    if (E->trace.ic_method==2)
        {
            fprintf(E->trace.fpt,"Reading individual tracer files\n");
        }

    fprintf(E->trace.fpt,"Number of tracer flavors: %d\n", E->trace.nflavors);

    if (E->trace.nflavors && E->trace.ic_method==0) {
        fprintf(E->trace.fpt,"Initialized tracer flavors by: %d\n", E->trace.ic_method_for_flavors);
        if (E->trace.ic_method_for_flavors == 0) {
            fprintf(E->trace.fpt,"Layered tracer flavors\n");
            for (i=0; i<E->trace.nflavors-1; i++)
                fprintf(E->trace.fpt,"Interface Height: %d %f\n",i,E->trace.z_interface[i]);
        }
#ifdef USE_GGRD
	else if(E->trace.ic_method_for_flavors == 1) {
            fprintf(E->trace.fpt,"netcdf grd assigned tracer flavors\n");
            fprintf(E->trace.fpt,"file: %s top %i layeres\n",E->trace.ggrd_file,
		    E->trace.ggrd_layers);
	}
#endif
        else {
            fprintf(E->trace.fpt,"Sorry-This IC methods for Flavors are Unavailable %d\n",E->trace.ic_method_for_flavors);
            fflush(E->trace.fpt);
            parallel_process_termination();
        }
    }

    for (i=0; i<E->trace.nflavors-2; i++) {
        if (E->trace.z_interface[i] < E->trace.z_interface[i+1]) {
            fprintf(E->trace.fpt,"Sorry - The %d-th z_interface is smaller than the next one.\n", i);
            fflush(E->trace.fpt);
            parallel_process_termination();
        }
    }



    /* regular grid stuff */

    fprintf(E->trace.fpt,"Regular Grid-> deltheta: %f delphi: %f\n",
            E->trace.deltheta[0],E->trace.delphi[0]);




    /* more obscure stuff */

    fprintf(E->trace.fpt,"Box Cushion: %f\n",E->trace.box_cushion);
    fprintf(E->trace.fpt,"Number of Basic Quantities: %d\n",
            E->trace.number_of_basic_quantities);
    fprintf(E->trace.fpt,"Number of Extra Quantities: %d\n",
            E->trace.number_of_extra_quantities);
    fprintf(E->trace.fpt,"Total Number of Tracer Quantities: %d\n",
            E->trace.number_of_tracer_quantities);


    /* analytical test */

    if (E->trace.ianalytical_tracer_test==1)
        {
            fprintf(E->trace.fpt,"\n\n ! Analytical Test Being Performed ! \n");
            fprintf(E->trace.fpt,"(some of the above parameters may not be used or applied\n");
            fprintf(E->trace.fpt,"Velocity functions given in main code\n");
            fflush(E->trace.fpt);
        }

    if (E->trace.itracer_warnings==0)
        {
            fprintf(E->trace.fpt,"\n WARNING EXITS ARE TURNED OFF! TURN THEM ON!\n");
            fprintf(stderr,"\n WARNING EXITS ARE TURNED OFF! TURN THEM ON!\n");
            fflush(E->trace.fpt);
        }

    write_composition_instructions(E);
    return;
}


/*********  ICHECK COLUMN NEIGHBORS ***************************/
/*                                                            */
/* This function check whether a point is in a neighboring    */
/* column. Neighbor surface element number is returned        */

static int icheck_column_neighbors(struct All_variables *E,
                                   int j, int nel,
                                   double x, double y, double z,
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
                    ival=icheck_element_column(E,j,neighbor[kk],x,y,z,rad);
                    if (ival>0)
                        {
                            return neighbor[kk];
                        }
                }
        }

    return -99;
}


/********** ICHECK ALL COLUMNS ********************************/
/*                                                            */
/* This function check all columns until the proper one for   */
/* a point (x,y,z) is found. The surface element is returned  */
/* else -99 if can't be found.                                */

static int icheck_all_columns(struct All_variables *E,
                              int j,
                              double x, double y, double z,
                              double rad)
{

    int icheck;
    int nel;

    int elz=E->lmesh.elz;
    int numel=E->lmesh.nel;

    for (nel=elz;nel<=numel;nel=nel+elz)
        {
            icheck=icheck_element_column(E,j,nel,x,y,z,rad);
            if (icheck==1)
                {
                    return nel;
                }
        }


    return -99;
}


/******* ICHECK ELEMENT *************************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given element                                          */

static int icheck_element(struct All_variables *E,
                          int j, int nel,
                          double x, double y, double z,
                          double rad)
{

    int icheck;

    icheck=icheck_shell(E,nel,rad);
    if (icheck==0)
        {
            return 0;
        }

    icheck=icheck_element_column(E,j,nel,x,y,z,rad);
    if (icheck==0)
        {
            return 0;
        }


    return 1;
}


/********  ICHECK SHELL ************************************/
/*                                                         */
/* This function serves to check whether a point lies      */
/* within the proper radial shell of a given element       */
/* note: j set to 1; shouldn't depend on cap               */

static int icheck_shell(struct All_variables *E,
                        int nel, double rad)
{

    int ival;
    int ibottom_node;
    int itop_node;

    double bottom_rad;
    double top_rad;


    ibottom_node=E->ien[1][nel].node[1];
    itop_node=E->ien[1][nel].node[5];

    bottom_rad=E->sx[1][3][ibottom_node];
    top_rad=E->sx[1][3][itop_node];

    ival=0;
    if ((rad>=bottom_rad)&&(rad<top_rad)) ival=1;

    return ival;
}

/********  ICHECK ELEMENT COLUMN ****************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given element's column                                 */

static int icheck_element_column(struct All_variables *E,
                                 int j, int nel,
                                 double x, double y, double z,
                                 double rad)
{

    double test_point[4];
    double rnode[5][10];

    int lev = E->mesh.levmax;
    int ival;
    int kk;
    int node;


    E->trace.istat_elements_checked++;

    /* surface coords of element nodes */

    for (kk=1;kk<=4;kk++)
        {

            node=E->ien[j][nel].node[kk+4];

            rnode[kk][1]=E->x[j][1][node];
            rnode[kk][2]=E->x[j][2][node];
            rnode[kk][3]=E->x[j][3][node];

            rnode[kk][4]=E->sx[j][1][node];
            rnode[kk][5]=E->sx[j][2][node];

            rnode[kk][6]=E->SinCos[lev][j][2][node]; /* cos(theta) */
            rnode[kk][7]=E->SinCos[lev][j][0][node]; /* sin(theta) */
            rnode[kk][8]=E->SinCos[lev][j][3][node]; /* cos(phi) */
            rnode[kk][9]=E->SinCos[lev][j][1][node]; /* sin(phi) */

        }

    /* test_point - project to outer radius */

    test_point[1]=x/rad;
    test_point[2]=y/rad;
    test_point[3]=z/rad;

    ival=icheck_bounds(E,test_point,rnode[1],rnode[2],rnode[3],rnode[4]);


    return ival;
}


/********* ICHECK CAP ***************************************/
/*                                                          */
/* This function serves to determine if a point lies within */
/* a given cap                                              */
/*                                                          */
int full_icheck_cap(struct All_variables *E, int icap,
                    double x, double y, double z, double rad)
{

    double test_point[4];
    double rnode[5][10];

    int ival;
    int kk;

    /* surface coords of cap nodes */


    for (kk=1;kk<=4;kk++)
        {

            rnode[kk][1]=E->trace.xcap[icap][kk];
            rnode[kk][2]=E->trace.ycap[icap][kk];
            rnode[kk][3]=E->trace.zcap[icap][kk];
            rnode[kk][4]=E->trace.theta_cap[icap][kk];
            rnode[kk][5]=E->trace.phi_cap[icap][kk];
            rnode[kk][6]=E->trace.cos_theta[icap][kk];
            rnode[kk][7]=E->trace.sin_theta[icap][kk];
            rnode[kk][8]=E->trace.cos_phi[icap][kk];
            rnode[kk][9]=E->trace.sin_phi[icap][kk];
        }


    /* test_point - project to outer radius */

    test_point[1]=x/rad;
    test_point[2]=y/rad;
    test_point[3]=z/rad;

    ival=icheck_bounds(E,test_point,rnode[1],rnode[2],rnode[3],rnode[4]);


    return ival;
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
                         double *test_point,
                         double *rnode1, double *rnode2,
                         double *rnode3, double *rnode4)
{

    int number_of_tries=0;
    int icheck;

    double v12[4];
    double v23[4];
    double v34[4];
    double v41[4];
    double v1p[4];
    double v2p[4];
    double v3p[4];
    double v4p[4];
    double cross1[4];
    double cross2[4];
    double cross3[4];
    double cross4[4];
    double rad1,rad2,rad3,rad4;
    double theta, phi;
    double tiny, eps;
    double x,y,z;

    double myatan();

    /* make vectors from node to node */

    makevector(v12,rnode2[1],rnode2[2],rnode2[3],rnode1[1],rnode1[2],rnode1[3]);
    makevector(v23,rnode3[1],rnode3[2],rnode3[3],rnode2[1],rnode2[2],rnode2[3]);
    makevector(v34,rnode4[1],rnode4[2],rnode4[3],rnode3[1],rnode3[2],rnode3[3]);
    makevector(v41,rnode1[1],rnode1[2],rnode1[3],rnode4[1],rnode4[2],rnode4[3]);

 try_again:

    number_of_tries++;

    /* make vectors from test point to node */

    makevector(v1p,test_point[1],test_point[2],test_point[3],rnode1[1],rnode1[2],rnode1[3]);
    makevector(v2p,test_point[1],test_point[2],test_point[3],rnode2[1],rnode2[2],rnode2[3]);
    makevector(v3p,test_point[1],test_point[2],test_point[3],rnode3[1],rnode3[2],rnode3[3]);
    makevector(v4p,test_point[1],test_point[2],test_point[3],rnode4[1],rnode4[2],rnode4[3]);

    /* Calculate cross products */

    crossit(cross2,v12,v2p);
    crossit(cross3,v23,v3p);
    crossit(cross4,v34,v4p);
    crossit(cross1,v41,v1p);

    /* Calculate radial component of cross products */

    rad1=findradial(E,cross1,rnode1[6],rnode1[7],rnode1[8],rnode1[9]);
    rad2=findradial(E,cross2,rnode2[6],rnode2[7],rnode2[8],rnode2[9]);
    rad3=findradial(E,cross3,rnode3[6],rnode3[7],rnode3[8],rnode3[9]);
    rad4=findradial(E,cross4,rnode4[6],rnode4[7],rnode4[8],rnode4[9]);

    /*  Check if any radial components is zero (along a boundary), adjust if so */
    /*  Hopefully, this doesn't happen often, may be expensive                  */

    tiny=1e-15;
    eps=1e-6;

    if (number_of_tries>3)
        {
            fprintf(E->trace.fpt,"Error(icheck_bounds)-too many tries\n");
            fprintf(E->trace.fpt,"Rads: %f %f %f %f\n",rad1,rad2,rad3,rad4);
            fprintf(E->trace.fpt,"Test Point: %f %f %f  \n",test_point[1],test_point[2],test_point[3]);
            fprintf(E->trace.fpt,"Nodal points: 1: %f %f %f\n",rnode1[1],rnode1[2],rnode1[3]);
            fprintf(E->trace.fpt,"Nodal points: 2: %f %f %f\n",rnode2[1],rnode2[2],rnode2[3]);
            fprintf(E->trace.fpt,"Nodal points: 3: %f %f %f\n",rnode3[1],rnode3[2],rnode3[3]);
            fprintf(E->trace.fpt,"Nodal points: 4: %f %f %f\n",rnode4[1],rnode4[2],rnode4[3]);
            fflush(E->trace.fpt);
            exit(10);
        }

    if (fabs(rad1)<=tiny||fabs(rad2)<=tiny||fabs(rad3)<=tiny||fabs(rad4)<=tiny)
        {
            x=test_point[1];
            y=test_point[2];
            z=test_point[3];
            theta=myatan(sqrt(x*x+y*y),z);
            phi=myatan(y,x);

            if (theta<=M_PI/2.0)
                {
                    theta=theta+eps;
                }
            else
                {
                    theta=theta-eps;
                }
            phi=phi+eps;
            x=sin(theta)*cos(phi);
            y=sin(theta)*sin(phi);
            z=cos(theta);
            test_point[1]=x;
            test_point[2]=y;
            test_point[3]=z;

            number_of_tries++;
            goto try_again;

        }

    icheck=0;
    if (rad1>0.0&&rad2>0.0&&rad3>0.0&&rad4>0.0) icheck=1;

    /*
      fprintf(stderr,"%d: icheck: %d\n",E->parallel.me,icheck);
      fprintf(stderr,"%d: rads: %f %f %f %f\n",E->parallel.me,rad1,rad2,rad3,rad4);
    */

    return icheck;

}

/****************************************************************************/
/* FINDRADIAL                                                              */
/*                                                                          */
/* This function finds the radial component of a Cartesian vector           */

static double findradial(struct All_variables *E, double *vec,
                         double cost, double sint,
                         double cosf, double sinf)
{
    double radialparti,radialpartj,radialpartk;
    double radial;

    radialparti=vec[1]*sint*cosf;
    radialpartj=vec[2]*sint*sinf;
    radialpartk=vec[3]*cost;

    radial=radialparti+radialpartj+radialpartk;


    return radial;
}


/******************MAKEVECTOR*********************************************************/

static void makevector(double *vec, double xf, double yf, double zf,
                       double x0, double y0, double z0)
{

    vec[1]=xf-x0;
    vec[2]=yf-y0;
    vec[3]=zf-z0;


    return;
}

/********************CROSSIT********************************************************/

static void crossit(double *cross, double *A, double *B)
{

    cross[1]=A[2]*B[3]-A[3]*B[2];
    cross[2]=A[3]*B[1]-A[1]*B[3];
    cross[3]=A[1]*B[2]-A[2]*B[1];


    return;
}


/************ FIX RADIUS ********************************************/
/* This function moves particles back in bounds if they left     */
/* during advection                                              */

static void fix_radius(struct All_variables *E,
                       double *radius, double *theta, double *phi,
                       double *x, double *y, double *z)
{
    double sint,cost,sinf,cosf,rad;
    double max_radius, min_radius;

    max_radius = E->sphere.ro - E->trace.box_cushion;
    min_radius = E->sphere.ri + E->trace.box_cushion;

    if (*radius > max_radius) {
        *radius=max_radius;
        rad=max_radius;
        cost=cos(*theta);
        sint=sqrt(1.0-cost*cost);
        cosf=cos(*phi);
        sinf=sin(*phi);
        *x=rad*sint*cosf;
        *y=rad*sint*sinf;
        *z=rad*cost;
    }
    if (*radius < min_radius) {
        *radius=min_radius;
        rad=min_radius;
        cost=cos(*theta);
        sint=sqrt(1.0-cost*cost);
        cosf=cos(*phi);
        sinf=sin(*phi);
        *x=rad*sint*cosf;
        *y=rad*sint*sinf;
        *z=rad*cost;
    }

    return;
}


/******************************************************************/
/* FIX PHI                                                        */
/*                                                                */
/* This function constrains the value of phi to be                */
/* between 0 and 2 PI                                             */
/*                                                                */

static void fix_phi(double *phi)
{
    const double two_pi=2.0*M_PI;

    double d2 = floor(*phi / two_pi);

    *phi -= two_pi * d2;

    return;
}

/******************************************************************/
/* FIX THETA PHI                                                  */
/*                                                                */
/* This function constrains the value of theta to be              */
/* between 0 and  PI, and                                         */
/* this function constrains the value of phi to be                */
/* between 0 and 2 PI                                             */
/*                                                                */
static void fix_theta_phi(double *theta, double *phi)
{
    const double two_pi=2.0*M_PI;
    double d, d2;

    d = floor(*theta/M_PI);

    *theta -= M_PI * d;
    *phi += M_PI * d;

    d2 = floor(*phi / two_pi);

    *phi -= two_pi * d2;

    return;
}

/********** IGET ELEMENT *****************************************/
/*                                                               */
/* This function returns the the real element for a given point. */
/* Returns -99 if not in this cap.                               */
/* iprevious_element, if known, is the last known element. If    */
/* it is not known, input a negative number.                     */

int full_iget_element(struct All_variables *E,
                      int j, int iprevious_element,
                      double x, double y, double z,
                      double theta, double phi, double rad)
{
    int icheck_processor_shell();
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
            ival=icheck_processor_shell(E,j,rad);
            if (ival!=1) return -99;
        }

    /* do quick search to see if element can be easily found. */
    /* note that element may still be out of this cap, but    */
    /* it is probably fast to do a quick search before        */
    /* checking cap                                           */


    /* get regular element number */

    iregel=iget_regel(E,j,theta,phi,&ntheta,&nphi);
    if (iregel<=0)
        {
            return -99;
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
            ival=icheck_element_column(E,j,iprevious_element,x,y,z,rad);
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
                            ival=icheck_element_column(E,j,nelem,x,y,z,rad);
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
            iel=icheck_column_neighbors(E,j,iprevious_element,x,y,z,rad);
            if (iel>0)
                {
                    goto foundit;
                }
        }

    /* check if still in cap */

    ival=full_icheck_cap(E,0,x,y,z,rad);
    if (ival==0)
        {
            return -99;
        }

    /* if here, still in cap (hopefully, without a doubt) */

    /* check cap corners (they are sometimes tricky) */

    elxz=elx*elz;
    icorner[1]=elz;
    icorner[2]=elxz;
    icorner[3]=elxz*(ely-1)+elz;
    icorner[4]=elxz*ely;
    for (kk=1;kk<=4;kk++)
        {
            ival=icheck_element_column(E,j,icorner[kk],x,y,z,rad);
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
                            iel=icheck_column_neighbors(E,j,ineighbor,x,y,z,rad);
                            if (iel>0)
                                {
                                    goto foundit;
                                }
                        }
                }

        }

    /* As a last resort, check all element columns */

    E->trace.istat1++;

    iel=icheck_all_columns(E,j,x,y,z,rad);

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
            x,y,z,theta,phi,iregel);
    fflush(E->trace.fpt);
    exit(10);

 foundit:

    /* find radial element */

    ifinal_iel=iget_radial_element(E,j,iel,rad);

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
/* exists. If not found, returns -99.                            */
/* npi and ntheta are modified for later use                     */

static int iget_regel(struct All_variables *E, int j,
                      double theta, double phi,
                      int *ntheta, int *nphi)
{

    int iregel;
    int idum;

    double rdum;

    /* first check whether theta is in range */

    if (theta<E->trace.thetamin[j]) return -99;
    if (theta>E->trace.thetamax[j]) return -99;

    /* get ntheta, nphi on regular mesh */

    rdum=theta-E->trace.thetamin[j];
    idum=rdum/E->trace.deltheta[j];
    *ntheta=idum+1;

    rdum=phi-E->trace.phimin[j];
    idum=rdum/E->trace.delphi[j];
    *nphi=idum+1;

    iregel=*ntheta+(*nphi-1)*E->trace.numtheta[j];

    /* check range to be sure */

    if (iregel>E->trace.numregel[j]) return -99;
    if (iregel<1) return -99;

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

static void define_uv_space(struct All_variables *E)
{
    const int j = 1;
    const int lev = E->mesh.levmax;
    const int refnode = 1;
    int i, n;

    double u, v, cosc, theta_f, phi_f, dphi, cosd;
    double *cost, *sint, *cosf, *sinf;

    if ((E->gnomonic = malloc((E->lmesh.nsf+1)*sizeof(struct GNOMONIC)))
        == NULL) {
        fprintf(stderr,"Error(define uv)-not enough memory(a)\n");
        exit(10);
    }

    sint = E->SinCos[lev][j][0];
    sinf = E->SinCos[lev][j][1];
    cost = E->SinCos[lev][j][2];
    cosf = E->SinCos[lev][j][3];

    /* uv space requires a reference point, using the 1st node */

    theta_f = E->sx[j][1][refnode];
    phi_f = E->sx[j][2][refnode];

    /* store cos(theta_f) and sin(theta_f) */
    E->gnomonic[0].u = cost[refnode];
    E->gnomonic[0].v = sint[refnode];

    /* convert each nodal point to u and v */

    for (i=1, n=1; i<=E->lmesh.nsf; i++, n+=E->lmesh.noz) {
        dphi = E->sx[j][2][n] - phi_f;
        cosd = cos(dphi);
        cosc = cost[refnode] * cost[n] + sint[refnode] * sint[n] * cosd;
        u = sint[n] * sin(dphi) / cosc;
        v = (sint[refnode] * cost[n] - cost[refnode] * sint[n] * cosd)
            / cosc;

        E->gnomonic[i].u = u;
        E->gnomonic[i].v = v;
    }

    return;
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

static void determine_shape_coefficients(struct All_variables *E)
{
    const int j = 1;
    int nelem, iwedge, kk, i;
    int snode;

    double u[5], v[5];
    double x1 = 0.0;
    double x2 = 0.0;
    double x3 = 0.0;
    double y1 = 0.0;
    double y2 = 0.0;
    double y3 = 0.0;
    double delta, a0, a1, a2;

    /* first, allocate memory */

    for(iwedge=1; iwedge<=2; iwedge++) {
        for (kk=1; kk<=9; kk++) {
            if ((E->trace.shape_coefs[j][iwedge][kk] =
                 (double *)malloc((E->lmesh.snel+1)*sizeof(double))) == NULL) {
                fprintf(E->trace.fpt,"ERROR(find shape coefs)-not enough memory(a)\n");
                fflush(E->trace.fpt);
                exit(10);
            }
        }
    }

    for (i=1, nelem=1; i<=E->lmesh.snel; i++, nelem+=E->lmesh.elz) {

        /* find u,v of local nodes at one radius  */

        for(kk=1; kk<=4; kk++) {
            snode = (E->ien[j][nelem].node[kk]-1) / E->lmesh.noz + 1;
            u[kk] = E->gnomonic[snode].u;
            v[kk] = E->gnomonic[snode].v;
        }

        for(iwedge=1; iwedge<=2; iwedge++) {

            if (iwedge == 1) {
                x1 = u[1];
                x2 = u[2];
                x3 = u[3];
                y1 = v[1];
                y2 = v[2];
                y3 = v[3];
            }
            if (iwedge == 2) {
                x1 = u[1];
                x2 = u[3];
                x3 = u[4];
                y1 = v[1];
                y2 = v[3];
                y3 = v[4];
            }

            /* shape function 1 */

            delta = (x3-x2)*(y1-y2)-(y2-y3)*(x2-x1);
            a0 = (x2*y3-x3*y2)/delta;
            a1 = (y2-y3)/delta;
            a2 = (x3-x2)/delta;

            E->trace.shape_coefs[j][iwedge][1][i] = a0;
            E->trace.shape_coefs[j][iwedge][2][i] = a1;
            E->trace.shape_coefs[j][iwedge][3][i] = a2;

            /* shape function 2 */

            delta = (x3-x1)*(y2-y1)-(y1-y3)*(x1-x2);
            a0 = (x1*y3-x3*y1)/delta;
            a1 = (y1-y3)/delta;
            a2 = (x3-x1)/delta;

            E->trace.shape_coefs[j][iwedge][4][i] = a0;
            E->trace.shape_coefs[j][iwedge][5][i] = a1;
            E->trace.shape_coefs[j][iwedge][6][i] = a2;

            /* shape function 3 */

            delta = (x1-x2)*(y3-y2)-(y2-y1)*(x2-x3);
            a0 = (x2*y1-x1*y2)/delta;
            a1 = (y2-y1)/delta;
            a2 = (x1-x2)/delta;

            E->trace.shape_coefs[j][iwedge][7][i] = a0;
            E->trace.shape_coefs[j][iwedge][8][i] = a1;
            E->trace.shape_coefs[j][iwedge][9][i] = a2;

        } /* end wedge */
    } /* end elem */

    return;
}


/*********** KEEP WITHIN BOUNDS *****************************************/
/*                                                                      */
/* This function makes sure the particle is within the sphere, and      */
/* phi and theta are within the proper degree range.                    */

void full_keep_within_bounds(struct All_variables *E,
                             double *x, double *y, double *z,
                             double *theta, double *phi, double *rad)
{
    fix_theta_phi(theta, phi);
    fix_radius(E,rad,theta,phi,x,y,z);

    return;
}


/* &&&&&&&&&&&&&&&&&&&& ANALYTICAL TESTS &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&**************/

/**************** ANALYTICAL TEST *********************************************************/
/*                                                                                        */
/* This function (and the 2 following) are used to test advection of tracers by assigning */
/* a test function (in "analytical_test_function").                                       */

void analytical_test(E)
     struct All_variables *E;

{
#if 0
    int kk,pp;
    int nsteps;
    int j;
    int my_number,number;
    int nrunge_steps;
    int nrunge_refinement;

    double dt;
    double runge_dt;
    double theta,phi,rad;
    double time;
    double vel_s[4];
    double vel_c[4];
    double my_theta0,my_phi0,my_rad0;
    double my_thetaf,my_phif,my_radf;
    double theta0,phi0,rad0;
    double thetaf,phif,radf;
    double x0_s[4],xf_s[4];
    double x0_c[4],xf_c[4];
    double vec[4];
    double runge_path_length,runge_time;
    double x0,y0,z0;
    double xf,yf,zf;
    double difference;
    double difperpath;

    void analytical_test_function();
    void predict_tracers();
    void correct_tracers();
    void analytical_runge_kutte();
    void sphere_to_cart();


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

                    analytical_test_function(E,theta,phi,rad,vel_s,vel_c);

                    E->sphere.cap[j].V[1][kk]=vel_s[1];
                    E->sphere.cap[j].V[2][kk]=vel_s[2];
                    E->sphere.cap[j].V[3][kk]=vel_s[3];
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
            if (E->trace.ntracers[j]>10)
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
            for (pp=1;pp<=E->trace.ntracers[j];pp++)
                {
                    theta=E->trace.basicq[j][0][pp];
                    phi=E->trace.basicq[j][1][pp];
                    rad=E->trace.basicq[j][2][pp];

                    fprintf(E->trace.fpt,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                    if (pp==1) fprintf(stderr,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                    if (pp==1)
                        {
                            my_theta0=theta;
                            my_phi0=phi;
                            my_rad0=rad;
                        }
                }
        }

    /* advect tracers */

    for (kk=1;kk<=nsteps;kk++)
        {
            E->monitor.solution_cycles=kk;

            time=time+dt;

            predict_tracers(E);
            correct_tracers(E);

            for (j=1;j<=E->sphere.caps_per_proc;j++)
                {
                    for (pp=1;pp<=E->trace.ntracers[j];pp++)
                        {
                            theta=E->trace.basicq[j][0][pp];
                            phi=E->trace.basicq[j][1][pp];
                            rad=E->trace.basicq[j][2][pp];

                            fprintf(E->trace.fpt,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                            if (pp==1) fprintf(stderr,"(%d) time: %f theta: %f phi: %f rad: %f\n",E->monitor.solution_cycles,time,theta,phi,rad);

                            if ((kk==nsteps) && (pp==1))
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
            my_number=E->trace.ntracers[j];
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

    x0_s[1]=theta0;
    x0_s[2]=phi0;
    x0_s[3]=rad0;

    nrunge_refinement=1000;

    nrunge_steps=nsteps*nrunge_refinement;
    runge_dt=dt/(1.0*nrunge_refinement);


    analytical_runge_kutte(E,nrunge_steps,runge_dt,x0_s,x0_c,xf_s,xf_c,vec);

    runge_time=vec[1];
    runge_path_length=vec[2];

    /* initial coordinates - both citcom and RK */

    x0=x0_c[1];
    y0=x0_c[2];
    z0=x0_c[3];

    /* convert final citcom coords into cartesian */

    sphere_to_cart(E,thetaf,phif,radf,&xf,&yf,&zf);

    difference=sqrt((xf-xf_c[1])*(xf-xf_c[1])+(yf-xf_c[2])*(yf-xf_c[2])+(zf-xf_c[3])*(zf-xf_c[3]));

    difperpath=difference/runge_path_length;

    /* Print out results */

    fprintf(E->trace.fpt,"Citcom calculation: steps: %d  dt: %f\n",nsteps,dt);
    fprintf(E->trace.fpt,"  (nodes per cap: %d x %d x %d)\n",E->lmesh.nox,E->lmesh.noy,(E->lmesh.noz-1)*E->parallel.nprocz+1);
    fprintf(E->trace.fpt,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
    fprintf(E->trace.fpt,"                    final position: theta: %f phi: %f rad: %f\n", thetaf,phif,radf);
    fprintf(E->trace.fpt,"                    (final time: %f) \n",time );

    fprintf(E->trace.fpt,"\n\nRunge-Kutte calculation: steps: %d  dt: %g\n",nrunge_steps,runge_dt);
    fprintf(E->trace.fpt,"                    starting position: theta: %f phi: %f rad: %f\n", theta0,phi0,rad0);
    fprintf(E->trace.fpt,"                    final position: theta: %f phi: %f rad: %f\n",xf_s[1],xf_s[2],xf_s[3]);
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
            fprintf(stderr,"                    final position: theta: %f phi: %f rad: %f\n",xf_s[1],xf_s[2],xf_s[3]);
            fprintf(stderr,"                    path length: %f \n",runge_path_length );
            fprintf(stderr,"                    (final time: %f) \n",runge_time );

            fprintf(stderr,"\n\n Difference between Citcom and RK: %e  (diff per path length: %e)\n\n",difference,difperpath);

        }

    fflush(E->trace.fpt);
#endif
    return;
}

/*************** ANALYTICAL RUNGE KUTTE ******************/
/*                                                       */
void analytical_runge_kutte(E,nsteps,dt,x0_s,x0_c,xf_s,xf_c,vec)
     struct All_variables *E;
     int nsteps;
     double dt;
     double *x0_c;
     double *x0_s;
     double *xf_c;
     double *xf_s;
     double *vec;

{

    int kk;

    double x_0,y_0,z_0;
    double x_p,y_p,z_p;
    double x_c=0.0;
    double y_c=0.0;
    double z_c=0.0;
    double theta_0,phi_0,rad_0;
    double theta_p,phi_p,rad_p;
    double theta_c,phi_c,rad_c;
    double vel0_s[4],vel0_c[4];
    double velp_s[4],velp_c[4];
    double time;
    double path,dtpath;

    void sphere_to_cart();
    void cart_to_sphere();
    void analytical_test_function();

    theta_0=x0_s[1];
    phi_0=x0_s[2];
    rad_0=x0_s[3];

    sphere_to_cart(E,theta_0,phi_0,rad_0,&x_0,&y_0,&z_0);

    /* fill initial cartesian vector to send back */

    x0_c[1]=x_0;
    x0_c[2]=y_0;
    x0_c[3]=z_0;

    time=0.0;
    path=0.0;

    for (kk=1;kk<=nsteps;kk++)
        {

            /* get velocity at initial position */

            analytical_test_function(E,theta_0,phi_0,rad_0,vel0_s,vel0_c);

            /* Find predicted midpoint position */

            x_p=x_0+vel0_c[1]*dt*0.5;
            y_p=y_0+vel0_c[2]*dt*0.5;
            z_p=z_0+vel0_c[3]*dt*0.5;

            /* convert to spherical */

            cart_to_sphere(E,x_p,y_p,z_p,&theta_p,&phi_p,&rad_p);

            /* get velocity at predicted midpoint position */

            analytical_test_function(E,theta_p,phi_p,rad_p,velp_s,velp_c);

            /* Find corrected position using midpoint velocity */

            x_c=x_0+velp_c[1]*dt;
            y_c=y_0+velp_c[2]*dt;
            z_c=z_0+velp_c[3]*dt;

            /* convert to spherical */

            cart_to_sphere(E,x_c,y_c,z_c,&theta_c,&phi_c,&rad_c);

            /* compute path lenght */

            dtpath=sqrt((x_c-x_0)*(x_c-x_0)+(y_c-y_0)*(y_c-y_0)+(z_c-z_0)*(z_c-z_0));
            path=path+dtpath;

            time=time+dt;

            x_0=x_c;
            y_0=y_c;
            z_0=z_c;

            /* next time step */

        }

    /* fill final spherical and cartesian vectors to send back */

    xf_s[1]=theta_c;
    xf_s[2]=phi_c;
    xf_s[3]=rad_c;

    xf_c[1]=x_c;
    xf_c[2]=y_c;
    xf_c[3]=z_c;

    vec[1]=time;
    vec[2]=path;

    return;
}



/**************** ANALYTICAL TEST FUNCTION ******************/
/*                                                          */
/* vel_s[1] => velocity in theta direction                  */
/* vel_s[2] => velocity in phi direction                    */
/* vel_s[3] => velocity in radial direction                 */
/*                                                          */
/* vel_c[1] => velocity in x direction                      */
/* vel_c[2] => velocity in y direction                      */
/* vel_c[3] => velocity in z direction                      */

void analytical_test_function(E,theta,phi,rad,vel_s,vel_c)
     struct All_variables *E;
     double theta,phi,rad;
     double *vel_s;
     double *vel_c;

{

    double sint,sinf,cost,cosf;
    double v_theta,v_phi,v_rad;
    double vx,vy,vz;

    /* This is where the function is given in spherical */

    v_theta=50.0*rad*cos(phi);
    v_phi=100.0*rad*sin(theta);
    v_rad=25.0*rad;

    vel_s[1]=v_theta;
    vel_s[2]=v_phi;
    vel_s[3]=v_rad;

    /* Convert the function into cartesian */

    sint=sin(theta);
    sinf=sin(phi);
    cost=cos(theta);
    cosf=cos(phi);

    vx=v_theta*cost*cosf-v_phi*sinf+v_rad*sint*cosf;
    vy=v_theta*cost*sinf+v_phi*cosf+v_rad*sint*sinf;
    vz=-v_theta*sint+v_rad*cost;

    vel_c[1]=vx;
    vel_c[2]=vy;
    vel_c[3]=vz;

    return;
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
