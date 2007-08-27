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

#include <sys/file.h>
#include <unistd.h>
#include "element_definitions.h"
#include "global_defs.h"

/* Private function prototypes */
static void backup_file(const char *output_file);
static void write_sentinel(FILE *fp);
static void read_sentinel(FILE *fp, int me);
static void general_checkpoint(struct All_variables *E, FILE *fp);

static void general_checkpoint(struct All_variables *E, FILE *fp);
static void tracer_checkpoint(struct All_variables *E, FILE *fp);
static void composition_checkpoint(struct All_variables *E, FILE *fp);
static void energy_checkpoint(struct All_variables *E, FILE *fp);
static void momentum_checkpoint(struct All_variables *E, FILE *fp);

static void read_general_checkpoint(struct All_variables *E, FILE *fp);
static void read_tracer_checkpoint(struct All_variables *E, FILE *fp);
static void read_composition_checkpoint(struct All_variables *E, FILE *fp);
static void read_energy_checkpoint(struct All_variables *E, FILE *fp);
static void read_momentum_checkpoint(struct All_variables *E, FILE *fp);


void output_checkpoint(struct All_variables *E)
{
    char output_file[255];
    FILE *fp1;

    sprintf(output_file, "%s.chkpt.%d.%d", E->control.data_file,
            E->parallel.me, E->monitor.solution_cycles);

    /* Disable the backup since the filename is unique. */
    /* backup_file(output_file); */

    fp1 = fopen(output_file, "wb");

    /* checkpoint for general information */
    /* this must be the first to be checkpointed */
    general_checkpoint(E, fp1);

    /* checkpoint for tracer/composition */
    if(E->control.tracer) {
        tracer_checkpoint(E, fp1);

        if(E->composition.on)
            composition_checkpoint(E, fp1);
    }

    /* checkpoint for energy equation */
    energy_checkpoint(E, fp1);

    /* checkpoint for momentum equation */
    momentum_checkpoint(E, fp1);

    fclose(fp1);
    return;
}


void read_checkpoint(struct All_variables *E)
{
    void initialize_material(struct All_variables *E);

    char output_file[255];
    FILE *fp;

    /* open the checkpoint file */
    snprintf(output_file, 254, "%s.chkpt.%d.%d", E->control.old_P_file,
             E->parallel.me, E->monitor.solution_cycles_init);
    fp = fopen(output_file, "rb");
    if(fp == NULL) {
        fprintf(stderr, "Cannot open file: %s\n", output_file);
        exit(-1);
    }

    /* check mesh information in the checkpoint file */
    read_general_checkpoint(E, fp);

    /* init E->mat */
    initialize_material(E);

    /* read tracer/composition information in the checkpoint file */
    if(E->control.tracer) {
        read_tracer_checkpoint(E, fp);

        if(E->composition.on)
            read_composition_checkpoint(E, fp);
    }

    /* read energy information in the checkpoint file */
    read_energy_checkpoint(E, fp);

    /* read momentum information in the checkpoint file */
    read_momentum_checkpoint(E, fp);

    fclose(fp);
    return;
}


static void backup_file(const char *output_file)
{
    char bak_file[255];
    int ierr;

    /* check the existence of output_file */
    if(access(output_file, F_OK) == 0) {
        /* if exist, renamed it to back up */
        sprintf(bak_file, "%s.bak", output_file);
        ierr = rename(output_file, bak_file);
        if(ierr != 0) {
            fprintf(stderr, "Warning, cannot backup checkpoint files\n");
        }
    }

    return;
}


static void write_sentinel(FILE *fp)
{
    int a[4] = {0, 0, 0, 0};

    fwrite(a, sizeof(int), 4, fp);
}


static void read_sentinel(FILE *fp, int me)
{
    int i, a[4];
    char nonzero = 0;

    fread(a, sizeof(int), 4, fp);

    /* check whether a[i] are all zero */
    for(i=0; i<4; i++)
        nonzero |= a[i];

    if(nonzero) {
        fprintf(stderr, "Error in reading checkpoint file: wrong sentinel, "
                "me=%d\n", me);
        exit(-1);
    }

    return;
}


static void general_checkpoint(struct All_variables *E, FILE *fp)
{
    /* write mesh information */
    fwrite(&(E->lmesh.nox), sizeof(int), 1, fp);
    fwrite(&(E->lmesh.noy), sizeof(int), 1, fp);
    fwrite(&(E->lmesh.noz), sizeof(int), 1, fp);
    fwrite(&(E->parallel.nprocx), sizeof(int), 1, fp);
    fwrite(&(E->parallel.nprocy), sizeof(int), 1, fp);
    fwrite(&(E->parallel.nprocz), sizeof(int), 1, fp);
    fwrite(&(E->sphere.caps_per_proc), sizeof(int), 1, fp);

    /* write timing information */
    fwrite(&(E->monitor.solution_cycles), sizeof(int), 1, fp);
    fwrite(&(E->monitor.elapsed_time), sizeof(float), 1, fp);
    fwrite(&(E->advection.timestep), sizeof(float), 1, fp);
    fwrite(&(E->control.start_age), sizeof(float), 1, fp);

    return;
}


static void read_general_checkpoint(struct All_variables *E, FILE *fp)
{
    int tmp[7];
    double dtmp;

    /* read mesh information */
    fread(tmp, sizeof(int), 7, fp);

    if((tmp[0] != E->lmesh.nox) ||
       (tmp[1] != E->lmesh.noy) ||
       (tmp[2] != E->lmesh.noz) ||
       (tmp[3] != E->parallel.nprocx) ||
       (tmp[4] != E->parallel.nprocy) ||
       (tmp[5] != E->parallel.nprocz) ||
       (tmp[6] != E->sphere.caps_per_proc)) {

        fprintf(stderr, "Error in reading checkpoint file: mesh, me=%d\n",
                E->parallel.me);
        fprintf(stderr, "%d %d %d %d %d %d %d\n",
                tmp[0], tmp[1], tmp[2], tmp[3],
                tmp[4], tmp[5], tmp[6]);
        exit(-1);
    }

    /* read timing information */
    fread(&(E->monitor.solution_cycles), sizeof(int), 1, fp);
    fread(&(E->monitor.elapsed_time), sizeof(float), 1, fp);
    fread(&(E->advection.timestep), sizeof(float), 1, fp);
    fread(&(E->control.start_age), sizeof(float), 1, fp);

    E->advection.timesteps = E->monitor.solution_cycles;

    return;
}


static void tracer_checkpoint(struct All_variables *E, FILE *fp)
{
    int m, i;

    write_sentinel(fp);

    fwrite(&(E->trace.number_of_basic_quantities), sizeof(int), 1, fp);
    fwrite(&(E->trace.number_of_extra_quantities), sizeof(int), 1, fp);
    fwrite(&(E->trace.nflavors), sizeof(int), 1, fp);
    fwrite(&(E->trace.ilast_tracer_count), sizeof(int), 1, fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++)
        fwrite(&(E->trace.ntracers[m]), sizeof(int), 1, fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(i=0; i<E->trace.number_of_basic_quantities; i++) {
            fwrite(E->trace.basicq[m][i], sizeof(double),
                   E->trace.ntracers[m]+1, fp);
        }
        for(i=0; i<E->trace.number_of_extra_quantities; i++) {
            fwrite(E->trace.extraq[m][i], sizeof(double),
                   E->trace.ntracers[m]+1, fp);
        }
        fwrite(E->trace.ielement[m], sizeof(int),
               E->trace.ntracers[m]+1, fp);
        for(i=0; i<E->trace.nflavors; i++) {
            fwrite(E->trace.ntracer_flavor[m][i], sizeof(double),
                   E->lmesh.nel+1, fp);
        }
    }

    return;
}


static void read_tracer_checkpoint(struct All_variables *E, FILE *fp)
{
    void allocate_tracer_arrays();

    int m, i, itmp;

    read_sentinel(fp, E->parallel.me);

    fread(&itmp, sizeof(int), 1, fp);
    if (itmp != E->trace.number_of_basic_quantities) {
        fprintf(stderr, "Error in reading checkpoint file: tracer basicq, me=%d\n",
                E->parallel.me);
        fprintf(stderr, "%d\n", itmp);
        exit(-1);

    }

    fread(&itmp, sizeof(int), 1, fp);
    if (itmp != E->trace.number_of_extra_quantities) {
        fprintf(stderr, "Error in reading checkpoint file: tracer extraq, me=%d\n",
                E->parallel.me);
        fprintf(stderr, "%d\n", itmp);
        exit(-1);

    }

    fread(&itmp, sizeof(int), 1, fp);
    if (itmp != E->trace.nflavors) {
        fprintf(stderr, "Error in reading checkpoint file: tracer nflavors, me=%d\n",
                E->parallel.me);
        fprintf(stderr, "%d\n", itmp);
        exit(-1);

    }

    fread(&itmp, sizeof(int), 1, fp);
    E->trace.ilast_tracer_count = itmp;

    /* # of tracers, allocate memory */
    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        fread(&itmp, sizeof(int), 1, fp);
        allocate_tracer_arrays(E, m, itmp);
        E->trace.ntracers[m] = itmp;
    }

    /* read tracer data */
    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(i=0; i<E->trace.number_of_basic_quantities; i++) {
            fread(E->trace.basicq[m][i], sizeof(double),
                  E->trace.ntracers[m]+1, fp);
        }
        for(i=0; i<E->trace.number_of_extra_quantities; i++) {
            fread(E->trace.extraq[m][i], sizeof(double),
                  E->trace.ntracers[m]+1, fp);
        }
        fread(E->trace.ielement[m], sizeof(int),
              E->trace.ntracers[m]+1, fp);
        for(i=0; i<E->trace.nflavors; i++) {
            fread(E->trace.ntracer_flavor[m][i], sizeof(double),
                  E->lmesh.nel+1, fp);
        }
    }

    return;
}


static void composition_checkpoint(struct All_variables *E, FILE *fp)
{
    int i, m;

    write_sentinel(fp);

    fwrite(&(E->composition.ncomp), sizeof(int), 1, fp);
    fwrite(E->composition.bulk_composition, sizeof(double),
           E->composition.ncomp, fp);
    fwrite(E->composition.initial_bulk_composition, sizeof(double),
           E->composition.ncomp, fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(i=0; i<E->composition.ncomp; i++)
            fwrite(E->composition.comp_el[m][i], sizeof(double),
                   E->lmesh.nel+1, fp);
    }
    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(i=0; i<E->composition.ncomp; i++)
            fwrite(E->composition.comp_node[m][i], sizeof(double),
                   E->lmesh.nno+1, fp);
    }

    return;
}


static void read_composition_checkpoint(struct All_variables *E, FILE *fp)
{
    double tmp;
    int m, i, itmp;

    read_sentinel(fp, E->parallel.me);

    fread(&itmp, sizeof(int), 1, fp);
    if (itmp != E->composition.ncomp) {
        fprintf(stderr, "Error in reading checkpoint file: ncomp, me=%d\n",
                E->parallel.me);
        fprintf(stderr, "%d\n", itmp);
        exit(-1);
    }

    fread(E->composition.bulk_composition, sizeof(double),
          E->composition.ncomp, fp);

    fread(E->composition.initial_bulk_composition, sizeof(double),
          E->composition.ncomp, fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(i=0; i<E->composition.ncomp; i++)
            fread(E->composition.comp_el[m][i], sizeof(double),
                  E->lmesh.nel+1, fp);
    }

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        for(i=0; i<E->composition.ncomp; i++)
            fread(E->composition.comp_node[m][i], sizeof(double),
                  E->lmesh.nno+1, fp);
    }

    /* preventing uninitialized access */
    E->trace.istat_iempty = 0;

    for (i=0; i<E->composition.ncomp; i++) {
        E->composition.error_fraction[i] = E->composition.bulk_composition[i]
        / E->composition.initial_bulk_composition[i] - 1.0;
    }

    return;
}


static void energy_checkpoint(struct All_variables *E, FILE *fp)
{
    int m;

    write_sentinel(fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        fwrite(E->T[m], sizeof(double), E->lmesh.nno+1, fp);
        fwrite(E->Tdot[m], sizeof(double), E->lmesh.nno+1, fp);
    }

    return;
}


static void read_energy_checkpoint(struct All_variables *E, FILE *fp)
{
    int m;

    read_sentinel(fp, E->parallel.me);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        fread(E->T[m], sizeof(double), E->lmesh.nno+1, fp);
        fread(E->Tdot[m], sizeof(double), E->lmesh.nno+1, fp);
    }

    return;
}


static void momentum_checkpoint(struct All_variables *E, FILE *fp)
{
    int m, i;
    int lev = E->mesh.levmax;

    write_sentinel(fp);

    fwrite(&(E->monitor.vdotv), sizeof(float), 1, fp);
    fwrite(&(E->monitor.incompressibility), sizeof(float), 1, fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        /* Pressure at equation points and nodes */
        /* Writing E->NP instead of calling p_to_nodes() because p_to_nodes()
         * contains parallel communication */
        fwrite(E->P[m], sizeof(double), E->lmesh.npno+1, fp);
        fwrite(E->NP[m], sizeof(float), E->lmesh.nno+1, fp);

        /* velocity at equation points */
        fwrite(E->U[m], sizeof(double), E->lmesh.neq+2, fp);

        /* viscosity at quadrature points and node points */
        fwrite(E->EVI[lev][m], sizeof(float),
               (E->lmesh.nel+2)*vpoints[E->mesh.nsd], fp);
        fwrite(E->VI[lev][m], sizeof(float), E->lmesh.nno+2, fp);
    }

    return;
}


static void read_momentum_checkpoint(struct All_variables *E, FILE *fp)
{
    void v_from_vector();

    int m, i;
    int lev = E->mesh.levmax;

    read_sentinel(fp, E->parallel.me);

    fread(&(E->monitor.vdotv), sizeof(float), 1, fp);
    fread(&(E->monitor.incompressibility), sizeof(float), 1, fp);

    for(m=1; m<=E->sphere.caps_per_proc; m++) {
        /* Pressure at equation points and nodes */
        fread(E->P[m], sizeof(double), E->lmesh.npno+1, fp);
        fread(E->NP[m], sizeof(float), E->lmesh.nno+1, fp);

        /* velocity at equation points */
        fread(E->U[m], sizeof(double), E->lmesh.neq+2, fp);

        /* viscosity at quadrature points and node points */
        fread(E->EVI[lev][m], sizeof(float),
              (E->lmesh.nel+2)*vpoints[E->mesh.nsd], fp);
        fread(E->VI[lev][m], sizeof(float), E->lmesh.nno+2, fp);
    }

    /* update velocity array */
    v_from_vector(E);

    return;
}



