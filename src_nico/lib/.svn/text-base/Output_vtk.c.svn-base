/*
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
 *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */
/* Routine to process the output of the finite element cycles
   and to turn them into a coherent suite  files  */


#include <stdlib.h>
#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "output.h"

static void vts_file_header(struct All_variables *E, FILE *fp)
{

    const char format[] =
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"StructuredGrid\" version=\"0.1\">\n"
        "  <StructuredGrid WholeExtent=\"%s\">\n"
        "    <Piece Extent=\"%s\">\n";

    char extent[64], header[1024];

    snprintf(extent, 64, "%d %d %d %d %d %d",
             E->lmesh.exs, E->lmesh.exs + E->lmesh.elx,
             E->lmesh.eys, E->lmesh.eys + E->lmesh.ely,
             E->lmesh.ezs, E->lmesh.ezs + E->lmesh.elz);

    snprintf(header, 1024, format, extent, extent);

    fputs(header, fp);

    return;
}


static void vts_file_trailer(struct All_variables *E, FILE *fp)
{
    const char trailer[] =
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n";

    fputs(trailer, fp);

    return;
}


static void vtk_point_data_header(struct All_variables *E, FILE *fp)
{
    fputs("      <PointData Scalars=\"temperature\" Vectors=\"velocity\">\n", fp);


    return;
}


static void vtk_point_data_trailer(struct All_variables *E, FILE *fp)
{
    fputs("      </PointData>\n", fp);
    return;
}


static void vtk_cell_data_header(struct All_variables *E, FILE *fp)
{
    fputs("      <CellData>\n", fp);
    return;
}


static void vtk_cell_data_trailer(struct All_variables *E, FILE *fp)
{
    fputs("      </CellData>\n", fp);
    return;
}


static void vtk_output_temp(struct All_variables *E, FILE *fp)
{
    int i, j;

    fputs("        <DataArray type=\"Float32\" Name=\"temperature\" format=\"ascii\">\n", fp);

    for(j=1; j<=E->sphere.caps_per_proc; j++) {
        for(i=1; i<=E->lmesh.nno; i++) {
            fprintf(fp, "%.6e\n", E->T[j][i]);
        }
    }

    fputs("        </DataArray>\n", fp);
    return;
}


static void vtk_output_velo(struct All_variables *E, FILE *fp)
{
    int i, j;
    double sint, sinf, cost, cosf;
    float *V[4];
    const int lev = E->mesh.levmax;

    fputs("        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);

    for(j=1; j<=E->sphere.caps_per_proc; j++) {
        V[1] = E->sphere.cap[j].V[1];
        V[2] = E->sphere.cap[j].V[2];
        V[3] = E->sphere.cap[j].V[3];

        for(i=1; i<=E->lmesh.nno; i++) {
            sint = E->SinCos[lev][j][0][i];
            sinf = E->SinCos[lev][j][1][i];
            cost = E->SinCos[lev][j][2][i];
            cosf = E->SinCos[lev][j][3][i];

            fprintf(fp, "%.6e %.6e %.6e\n",
                    V[1][i]*cost*cosf - V[2][i]*sinf + V[3][i]*sint*cosf,
                    V[1][i]*cost*sinf + V[2][i]*cosf + V[3][i]*sint*sinf,
                    -V[1][i]*sint + V[3][i]*cost);
        }
    }

    fputs("        </DataArray>\n", fp);
    return;
}


static void vtk_output_visc(struct All_variables *E, FILE *fp)
{
    int i, j;
    int lev = E->mesh.levmax;

    fputs("        <DataArray type=\"Float32\" Name=\"viscosity\" format=\"ascii\">\n", fp);

    for(j=1; j<=E->sphere.caps_per_proc; j++) {
        for(i=1; i<=E->lmesh.nno; i++)
            fprintf(fp, "%.4e\n", E->VI[lev][j][i]);
    }

    fputs("        </DataArray>\n", fp);
    return;
}


static void vtk_output_coord(struct All_variables *E, FILE *fp)
{
    /* Output Cartesian coordinates as most VTK visualization softwares
       assume it. */
    int i, j;

    fputs("      <Points>\n", fp);
    fputs("        <DataArray type=\"Float32\" Name=\"coordinate\" NumberOfComponents=\"3\" format=\"ascii\">\n", fp);

    for(j=1; j<=E->sphere.caps_per_proc; j++) {
        for(i=1; i<=E->lmesh.nno; i++)
            fprintf(fp,"%.6e %.6e %.6e\n",
                    E->x[j][1][i],
                    E->x[j][2][i],
                    E->x[j][3][i]);
    }

    fputs("        </DataArray>\n", fp);
    fputs("      </Points>\n", fp);

    return;
}


/**********************************************************************/

void vtk_output(struct All_variables *E, int cycles)
{
    char output_file[255];
    FILE *fp;

    snprintf(output_file, 255, "%s.%d.step%d.vts",
             E->control.data_file, E->parallel.me, cycles);
    fp = output_open(output_file, "w");


    /* first, write volume data to vts file */
    vts_file_header(E, fp);

    /* write node-based field */
    vtk_point_data_header(E, fp);
    vtk_output_temp(E, fp);
    vtk_output_velo(E, fp);
    vtk_output_visc(E, fp);
    vtk_point_data_trailer(E, fp);

    /* write element-based field */
    vtk_cell_data_header(E, fp);
    /**/
    vtk_cell_data_trailer(E, fp);

    /* write coordinate */
    vtk_output_coord(E, fp);

    vts_file_trailer(E, fp);

    /* then, write other type of data */
    //vtk_output_surf_botm(E, );


    fclose(fp);

    return;
}
