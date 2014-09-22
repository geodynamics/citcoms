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
   and to turn them into a coherent suite of files  */


#include <math.h>
#include "element_definitions.h"
#include "global_defs.h"
#include "output.h"
#include "string.h"

#ifdef USE_GZDIR
#include "zlib.h"
#endif

#define CHUNK 16384

static void write_binary_array(int nn, float* array, FILE * f);
static void write_ascii_array(int nn, int perLine, float *array, FILE *fp);

static void vts_file_header(struct All_variables *E, FILE *fp)
{

    const char format[] =
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"StructuredGrid\" version=\"0.1\" compressor=\"vtkZLibDataCompressor\" byte_order=\"LittleEndian\">\n"
        "  <StructuredGrid WholeExtent=\"%s\">\n"
        "    <Piece Extent=\"%s\">\n";

    char extent[64], header[1024];

    snprintf(extent, 64, "%d %d %d %d %d %d",
             E->lmesh.ezs, E->lmesh.ezs + E->lmesh.elz,
             E->lmesh.exs, E->lmesh.exs + E->lmesh.elx,
             E->lmesh.eys, E->lmesh.eys + E->lmesh.ely);

    snprintf(header, 1024, format, extent, extent);

    fputs(header, fp);
}


static void vts_file_trailer(struct All_variables *E, FILE *fp)
{
    const char trailer[] =
        "    </Piece>\n"
        "  </StructuredGrid>\n"
        "</VTKFile>\n";

    fputs(trailer, fp);

}


static void vtk_point_data_header(struct All_variables *E, FILE *fp)
{
    fputs("      <PointData Scalars=\"temperature\" Vectors=\"velocity\">\n", fp);
}


static void vtk_point_data_trailer(struct All_variables *E, FILE *fp)
{
    fputs("      </PointData>\n", fp);
}


static void vtk_cell_data_header(struct All_variables *E, FILE *fp)
{
    fputs("      <CellData>\n", fp);
}


static void vtk_cell_data_trailer(struct All_variables *E, FILE *fp)
{
    fputs("      </CellData>\n", fp);
}


static void vtk_output_temp(struct All_variables *E, FILE *fp)
{
    int i;
    int nodes = E->sphere.caps_per_proc*E->lmesh.nno;
    float* floattemp = malloc(nodes*sizeof(float));

    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"temperature\" format=\"%s\">\n", E->output.vtk_format);

    for(i=0;i <= nodes;i++)
        floattemp[i] =  (float) *(E->T+i+1);

    if (strcmp(E->output.vtk_format,"binary") == 0) {
        write_binary_array(nodes,floattemp,fp);
    } else {
        write_ascii_array(nodes,1,floattemp,fp);
    }
    fputs("        </DataArray>\n", fp);
    free(floattemp);
}


static void vtk_output_velo(struct All_variables *E, FILE *fp)
{
    int i, j;
    int nodes=E->sphere.caps_per_proc*E->lmesh.nno;
    double sint, sinf, cost, cosf;
    float *V[4];
    const int lev = E->mesh.levmax;
    float* floatvel = malloc(nodes*3*sizeof(float));

    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"%s\">\n", E->output.vtk_format);

        V[1] = E->sphere.cap[CPPR].V[1];
        V[2] = E->sphere.cap[CPPR].V[2];
        V[3] = E->sphere.cap[CPPR].V[3];

        for(i=1; i<=E->lmesh.nno; i++) {
            sint = E->SinCos[lev][CPPR][0][i];
            sinf = E->SinCos[lev][CPPR][1][i];
            cost = E->SinCos[lev][CPPR][2][i];
            cosf = E->SinCos[lev][CPPR][3][i];

            floatvel[(((CPPR-1)*E->sphere.caps_per_proc)+i-1)*3+0] = (float)(V[1][i]*cost*cosf - V[2][i]*sinf + V[3][i]*sint*cosf);
            floatvel[(((CPPR-1)*E->sphere.caps_per_proc)+i-1)*3+1] = (float)(V[1][i]*cost*sinf + V[2][i]*cosf + V[3][i]*sint*sinf);
            floatvel[(((CPPR-1)*E->sphere.caps_per_proc)+i-1)*3+2] = (float)(-V[1][i]*sint + V[3][i]*cost);
        }

    if (strcmp(E->output.vtk_format, "binary") == 0)
        write_binary_array(nodes*3,floatvel,fp);
    else
        write_ascii_array(nodes*3,3,floatvel,fp);
    fputs("        </DataArray>\n", fp);

    free(floatvel);
}


static void vtk_output_visc(struct All_variables *E, FILE *fp)
{
    int nodes = E->sphere.caps_per_proc*E->lmesh.nno;
    int lev = E->mesh.levmax;

    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"viscosity\" format=\"%s\">\n", E->output.vtk_format);
        if (strcmp(E->output.vtk_format, "binary") == 0) {
            write_binary_array(nodes,&E->VI[lev][1],fp);
        } else {
            write_ascii_array(nodes,1,&E->VI[lev][1],fp);
        }

    fputs("        </DataArray>\n", fp);
}


static void vtk_output_coord(struct All_variables *E, FILE *fp)
{
    /* Output Cartesian coordinates as most VTK visualization softwares
       assume it. */
    int i, j;
    int nodes = E->sphere.caps_per_proc*E->lmesh.nno;
    float* floatpos = malloc(nodes*3*sizeof(float));

    fputs("      <Points>\n", fp);
    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"coordinate\" NumberOfComponents=\"3\" format=\"%s\">\n", E->output.vtk_format);

        for(i=1; i<=E->lmesh.nno; i++){
          floatpos[((CPPR-1)*E->lmesh.nno+i-1)*3] = (float)(E->x[CPPR][1][i]);
	        floatpos[((CPPR-1)*E->lmesh.nno+i-1)*3+1]=(float)(E->x[CPPR][2][i]);
	        floatpos[((CPPR-1)*E->lmesh.nno+i-1)*3+2]=(float)(E->x[CPPR][3][i]);
        }

    if (strcmp(E->output.vtk_format, "binary") == 0)
        write_binary_array(nodes*3,floatpos,fp);
    else
        write_ascii_array(nodes*3,3,floatpos,fp);
    fputs("        </DataArray>\n", fp);
    fputs("      </Points>\n", fp);
    free(floatpos);
}

static void vtk_output_stress(struct All_variables *E, FILE *fp)
{
    int nodes = E->sphere.caps_per_proc*E->lmesh.nno;
   /* for stress computation */
    void allocate_STD_mem();
    void compute_nodal_stress();
    void free_STD_mem();
    float *SXX,*SYY,*SXY,*SXZ,*SZY,*SZZ;
    float *divv,*vorv;

    /* those are sorted like stt spp srr stp str srp  */
    allocate_STD_mem(E, &SXX, &SYY, &SZZ, &SXY, &SXZ, &SZY, &divv, &vorv);
    compute_nodal_stress(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);
    free_STD_mem(E, SXX, SYY, SZZ, SXY, SXZ, SZY, divv, vorv);

    fprintf(fp, "        <DataArray type=\"Float32\" Name=\"stress\" NumberOfComponents=\"6\" format=\"%s\">\n", E->output.vtk_format);

    if (strcmp(E->output.vtk_format, "binary") == 0) {
        write_binary_array(nodes*6,&E->gstress[CPPR][1],fp);
    } else {
        write_ascii_array(nodes*6,6,&E->gstress[CPPR][1],fp);
    }

    fputs("        </DataArray>\n", fp);
}

static void vtk_output_comp_nd(struct All_variables *E, FILE *fp)
{
    int i, j, k;
    char name[255];
    int nodes = E->sphere.caps_per_proc*E->lmesh.nno;
    float* floatcompo = malloc (nodes*sizeof(float));

    for(k=0;k<E->composition.ncomp;k++) {
        fprintf(fp, "        <DataArray type=\"Float32\" Name=\"composition%d\" format=\"%s\">\n", k+1, E->output.vtk_format);

            for(i=1; i<=E->lmesh.nno; i++) {
                floatcompo[(CPPR-1)*E->lmesh.nno+i-1] = (float) (E->composition.comp_node[CPPR][k][i]);
	    }

        if (strcmp(E->output.vtk_format, "binary") == 0)
            write_binary_array(nodes,floatcompo,fp);
        else
            write_ascii_array(nodes,1,floatcompo,fp);
        fputs("        </DataArray>\n", fp);
    }
    free(floatcompo);
}


static void vtk_output_surf(struct All_variables *E,  FILE *fp, int cycles)
{
    int i, j, k;
    int nodes = E->sphere.caps_per_proc*E->lmesh.nno;
    char output_file[255];
    float* floattopo = malloc (nodes*sizeof(float));

    if((E->output.write_q_files == 0) || (cycles == 0) ||
      (cycles % E->output.write_q_files)!=0)
        heat_flux(E);
  /* else, the heat flux will have been computed already */

    if(E->control.use_cbf_topo){
        get_CBF_topo(E,E->slice.tpg,E->slice.tpgb);
    }
    else{
        get_STD_topo(E,E->slice.tpg,E->slice.tpgb,E->slice.divg,E->slice.vort,cycles);
    }

    fprintf(fp,"        <DataArray type=\"Float32\" Name=\"surface\" format=\"%s\">\n", E->output.vtk_format);

        for(i=1;i<=E->lmesh.nsf;i++){
            for(k=1;k<=E->lmesh.noz;k++){
                floattopo[(CPPR-1)*E->lmesh.nno + (i-1)*E->lmesh.noz + k-1] = 0.0;
            }

            if (E->parallel.me_loc[3]==E->parallel.nprocz-1) {

                /* choose either STD topo or pseudo-free-surf topo */
                if(E->control.pseudo_free_surf)
                floattopo[(CPPR-1)*E->lmesh.nno + i*E->lmesh.noz-1] = E->slice.freesurf[i];
                else
                floattopo[(CPPR-1)*E->lmesh.nno + i*E->lmesh.noz-1] = E->slice.tpg[i];

            }
        }

    if (strcmp(E->output.vtk_format, "binary") == 0)
        write_binary_array(nodes,floattopo,fp);
    else
        write_ascii_array(nodes,1,floattopo,fp);

    fputs("        </DataArray>\n", fp);
}


static void write_vtm(struct All_variables *E, int cycles)
{
    FILE *fp;
    char vtm_file[255];
    int n;

    const char header[] =
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"vtkMultiBlockDataSet\" version=\"1.0\" compressor=\"vtkZLibDataCompressor\" byte_order=\"LittleEndian\">\n"
        "  <vtkMultiBlockDataSet>\n";

    snprintf(vtm_file, 255, "%s.%d.vtm",
             E->control.data_file, cycles);
    fp = output_open(vtm_file, "w");
    fputs(header, fp);

    for(n=0; n<E->parallel.nproc; n++) {
        fprintf(fp, "    <DataSet index=\"%d\" file=\"%s.proc%d.%d.vts\"/>\n",
                n, E->control.data_prefix, n, cycles);
    }
    fputs("  </vtkMultiBlockDataSet>\n",fp);
    fputs("</VTKFile>",fp);

    fclose(fp);
}

static void write_visit(struct All_variables *E, int cycles)
{
    FILE *fp;
    char visit_file[255];
    int n;

    const char header[] = "!NBLOCKS %d\n";

    snprintf(visit_file, 255, "%s.%d.visit",
             E->control.data_file, cycles);
    fp = output_open(visit_file, "w");
    fprintf(fp, header, E->parallel.nproc);

    for(n=0; n<E->parallel.nproc; n++) {
        fprintf(fp, "%s.proc%d.%d.vts\n",
                E->control.data_prefix, n, cycles);
    }
    fclose(fp);
}

static void write_pvts(struct All_variables *E, int cycles)
{
    FILE *fp;
    char pvts_file[255];
    int i,j,k;
    snprintf(pvts_file, 255, "%s.%d.pvts",
             E->control.data_file,cycles);
    fp = output_open(pvts_file, "w");

    const char format[] =
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"PStructuredGrid\" version=\"0.1\" compressor=\"vtkZLibDataCompressor\" byte_order=\"LittleEndian\">\n"
        "  <PStructuredGrid WholeExtent=\"%s\" GhostLevel=\"#\">\n"
        "    <PPointData Scalars=\"temperature\" Vectors=\"velocity\">\n"
        "      <DataArray type=\"Float32\" Name=\"temperature\" format=\"%s\"/>\n"
        "      <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"%s\"/>\n"
        "      <DataArray type=\"Float32\" Name=\"viscosity\" format=\"%s\"/>\n";

    char extent[64], header[1024];

    snprintf(extent, 64, "%d %d %d %d %d %d",
        E->lmesh.ezs, E->lmesh.ezs + E->lmesh.elz*E->parallel.nprocz,
        E->lmesh.exs, E->lmesh.exs + E->lmesh.elx*E->parallel.nprocx,
        E->lmesh.eys, E->lmesh.eys + E->lmesh.ely*E->parallel.nprocy);

    snprintf(header, 1024, format, extent, E->output.vtk_format,
             E->output.vtk_format, E->output.vtk_format);
    fputs(header, fp);

    if (E->output.stress){
        fprintf(fp,"      <DataArray type=\"Float32\" Name=\"stress\" NumberOfComponents=\"6\" format=\"%s\"/>\n", E->output.vtk_format);
    }
    if (E->output.comp_nd && E->composition.on){
        fprintf(fp,"      <DataArray type=\"Float32\" Name=\"composition1\" format=\"%s\"/>\n", E->output.vtk_format);
    }
    if (E->output.surf){
        fprintf(fp,"      <DataArray type=\"Float32\" Name=\"surface\" format=\"%s\"/>\n", E->output.vtk_format);
    }

    fputs("    </PPointData>\n \n"
    "    <PCellData>\n"
    "    </PCellData>\n \n"
    "    <PPoints>\n"
    "      <DataArray type=\"Float32\" Name=\"coordinate\" NumberOfComponents=\"3\" format=\"binary\" />\n"
    "    </PPoints>\n", fp);

    for(i=0; i < E->parallel.nprocy;i++){
        for(j=0; j < E->parallel.nprocx;j++){
            for(k=0; k < E->parallel.nprocz;k++){
                fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s.proc%d.%d.vts\"/>\n",
                    (k%E->parallel.nprocz)*E->lmesh.elz,
                    (k%E->parallel.nprocz+1)*E->lmesh.elz,
                    (j%E->parallel.nprocx)*E->lmesh.elx, (j%E->parallel.nprocx+1)*E->lmesh.elx,
                    (i%E->parallel.nprocy)*E->lmesh.ely, (i%E->parallel.nprocy+1)*E->lmesh.ely,
                    E->control.data_prefix,
                    i*E->parallel.nprocx*E->parallel.nprocz+j*E->parallel.nprocz+k, cycles);
            }
        }
    }

    fputs("  </PStructuredGrid>\n",fp);
    fputs("</VTKFile>",fp);

    fclose(fp);
}

static void write_ascii_array(int nn, int perLine, float *array, FILE *fp)
{
    int i;

    switch (perLine) {
    case 1:
        for(i=0; i<nn; i++)
            fprintf(fp, "%.4e\n", array[i]);
        break;
    case 3:
        for(i=0; i < nn/3; i++)
            fprintf(fp,"%.4e %.4e %.4e\n",array[3*i],array[3*i+1],array[3*i+2]);
        break;
    case 6:
        for(i=0; i < nn/6; i++)
            fprintf(fp,"%.4e %.4e %.4e %.4e %.4e %.4e\n",
                    array[6*i],array[6*i+1],array[6*i+2],
                    array[6*i+3],array[6*i+4],array[6*i+5]);
        break;
    }
}

static void FloatToUnsignedChar(float * floatarray, int nn, unsigned char * chararray)
{
    /* simple float to unsigned chararray routine via union
    nn=length(intarray) chararray has to be BIG ENOUGH! */
    int i;
    union FloatToUnsignedChars
        {
            float input;
            unsigned char output[4];
        } floattransform;

    for (i=0; i<nn; i++){
        floattransform.input=floatarray[i];
        chararray[4*i]=floattransform.output[0];
        chararray[4*i+1]=floattransform.output[1];
        chararray[4*i+2]=floattransform.output[2];
        chararray[4*i+3]=floattransform.output[3];
    }
}

static void IntToUnsignedChar(int * intarray, int nn, unsigned char * chararray)
{
    /* simple int - to unsigned chararray routine via union
    nn=length(intarray) chararray has to be BIG ENOUGH! */
    int i;
    union IntToUnsignedChars
        {
            int input;
            unsigned char output[4];
        } inttransform;

    for (i=0; i<nn; i++){
        inttransform.input=intarray[i];
        chararray[4*i]=inttransform.output[0];
        chararray[4*i+1]=inttransform.output[1];
        chararray[4*i+2]=inttransform.output[2];
        chararray[4*i+3]=inttransform.output[3];
        }
}


static void zlibcompress(unsigned char* in, int nn, unsigned char** out, int *nn2)
/* function to compress "in" to "out" reducing size from nn to nn2 */
{
#ifdef USE_GZDIR
    int ntemp=0;

    /* in and out of z-stream */
    unsigned char inz[CHUNK];
    unsigned char outz[CHUNK];

    /* compression level */
    int level = Z_DEFAULT_COMPRESSION;
    int ret,flush;
    int i,j,k;

    /* zlib compression stream */
    z_stream strm;

    /* hope compressed data will be <= uncompressed */
    *out = malloc(sizeof(unsigned char)*nn);

    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    /* zlib init */
    ret = deflateInit(&strm, level);
    if (ret == Z_OK){
        i=0;     // position in "in" array
        do{
            j=0; // position in "inz"
            do{
                inz[j++]=in[i++];
            } while((j<CHUNK) && (i<nn)); // stopps if "inz"-buffer is full or "in" array empty
            strm.avail_in=j;              // set number of input chars

            flush = (i==nn) ? Z_FINISH : Z_NO_FLUSH; // done?
            strm.next_in = inz;           // set input buffer

            do{
                strm.avail_out = CHUNK;   // set number of max output chars
                strm.next_out = outz;     // set output buffer

                /* zlib compress */
                ret = deflate(&strm, flush);
                assert(ret != Z_STREAM_ERROR);

                /* zlib changed strm.avail_out=CHUNK
                 to the number of chars we can NOT use
                 in outz */

                for (k=0;k<CHUNK-strm.avail_out;k++){
                    (*out)[ntemp+k]=outz[k];
                }

                /* increase position in "out" */
                ntemp+=(CHUNK-strm.avail_out);
            }while(strm.avail_out==0);
            assert(strm.avail_in == 0);

        }while (flush != Z_FINISH);
    }
    else{fprintf(stderr,"Error during compression init\n");}

    // now we know how short "out" should be!
    *nn2=ntemp;
    *out = realloc(*out,sizeof(unsigned char)*ntemp);

    (void)deflateEnd(&strm);
#endif

}

static void base64(unsigned char * in, int nn, unsigned char* out)
{
    /*takes *in*-array and "in"-length-"nn" and fills "out"-array
    with base64(in) "out" needs to be big enough!!!
    length(out) >= 4* |^ nn/3.0 ^| */
    char cb64[]="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    int len;
    int i;

    for (i=0; i < nn; i+=3){

        len = (3 < nn-i ? 3 : nn-i);
        if (len >= 3){
        /* normal base64 encoding */
            out[i/3*4+0] = cb64[ in[i] >> 2 ];
            out[i/3*4+1] = cb64[ ((in[i] & 0x03) << 4) | ((in[i+1] & 0xf0) >> 4) ];
            out[i/3*4+2] = cb64[ ((in[i+1] & 0x0f) << 2) | ((in[i+2] & 0xc0) >> 6)];
            out[i/3*4+3] = cb64[ in[i+2] & 0x3f ];
        } else if (len == 2){
        /* at the end of array fill up with '=' */
            out[i/3*4+0] = cb64[ in[i] >> 2 ];
            out[i/3*4+1] = cb64[ ((in[i] & 0x03) << 4) | ((in[i+1] & 0xf0) >> 4) ];
            out[i/3*4+2] = cb64[((in[i+1] & 0x0f) << 2)];
            out[i/3*4+3] = (unsigned char) '=';
        } else if (len == 1){
        /* at the end of array fill up with '=' */
            out[i/3*4+0] = cb64[ in[i] >> 2 ];
            out[i/3*4+1] = cb64[ ((in[i] & 0x03) << 4) ];
            out[i/3*4+2] = (unsigned char) '=';
            out[i/3*4+3] = (unsigned char) '=';
        }
    }
}


static void base64plushead(unsigned char * in, int nn, int orinn, unsigned char* out)
{
    /* writing vtk compatible zlib compressed base64 encoded data to "out" */
    int i;
    unsigned char * b64head;
    int b64bodylength;
    unsigned char * b64body;
    /* header of data */
    unsigned char * charhead = malloc(sizeof(unsigned char)*16);
    /* - consists of "1" (number of pieces) */
    /* - original datalength in byte */
    /* - original datalength in byte */
    /* - new datalength after z-lib compression */
    int * headInts= malloc(sizeof(int)*4);
    headInts[0]=1;
    headInts[1]=orinn;
    headInts[2]=orinn;
    headInts[3]=nn;
    // transform to unsigned char
    IntToUnsignedChar(headInts,4,charhead);

    // base64: 16byte -> 24byte
    b64head =  malloc(sizeof(unsigned char)*24);
    // fills b64head
    base64(charhead, 16, b64head);

    // base64 data
    b64bodylength = 4*ceil((double) nn/3.0);
    b64body = malloc(sizeof(unsigned char)*b64bodylength);
    // writes base64 data to b64body
    base64(in,nn,b64body);

    // combines header and body
    for (i=0; i<24 ; i++){
        out[i]=b64head[i];
    }

    for (i=0; i<b64bodylength ; i++){
        out[24+i]=b64body[i];
    }

    if(b64body){free(b64body);}
    if(b64head){free(b64head);}
    if(headInts){free(headInts);}
    if(charhead){free(charhead);}
}

static void write_binary_array(int nn, float* array, FILE * f)
{
    /* writes vtk-data array of floats and performs zip and base64 encoding */
    int chararraylength=4*nn;	/* nn floats -> 4*nn unsigned chars */
    unsigned char * chararray = malloc (chararraylength * sizeof(unsigned char));
    int compressedarraylength = 0;
    unsigned char * compressedarray;
    unsigned char ** pointertocompressedarray= &compressedarray;
    int base64plusheadlength;
    unsigned char * base64plusheadarray;

    FloatToUnsignedChar(array,nn,chararray);

    /* compression routine */
    zlibcompress(chararray,chararraylength,pointertocompressedarray,&compressedarraylength);

    /* special header for zip compressed and bas64 encoded data
    header needs 4 int32 = 16 byte -> 24 byte due to base64 (4*16/3) */
    base64plusheadlength = 24 + 4*ceil((double) compressedarraylength/3.0);
    base64plusheadarray = malloc(sizeof(unsigned char)* base64plusheadlength);

    /* fills base64plusheadarray with everything ready for simple writing */
    base64plushead(compressedarray,compressedarraylength, chararraylength, base64plusheadarray);

    fwrite(base64plusheadarray,sizeof(unsigned char),base64plusheadlength,f);
    fprintf(f,"\n");
    free(chararray);
    free(base64plusheadarray);
    free(compressedarray);
}

/**********************************************************************/

void vtk_output(struct All_variables *E, int cycles)
{
    char output_file[255];
    FILE *fp;
    snprintf(output_file, 255, "%s.proc%d.%d.vts",
             E->control.data_file, E->parallel.me, cycles);
    fp = output_open(output_file, "w");

    /* first, write volume data to vts file */
    vts_file_header(E, fp);

    /* write node-based field */
    vtk_point_data_header(E, fp);
    vtk_output_temp(E, fp);

    vtk_output_velo(E, fp);

    vtk_output_visc(E, fp);

    if (E->output.stress)
        vtk_output_stress(E, fp);

    if (E->output.comp_nd && E->composition.on)
        vtk_output_comp_nd(E, fp);

    if (E->output.surf)
        vtk_output_surf(E, fp, cycles);

    vtk_point_data_trailer(E, fp);

    /* write element-based field */
    vtk_cell_data_header(E, fp);
    /* TODO: comp_el, heating */
    vtk_cell_data_trailer(E, fp);

    /* write coordinate */
    vtk_output_coord(E, fp);

    vts_file_trailer(E, fp);
    fclose(fp);

    /* then, write other type of data */

    if (E->parallel.me == 0) {
        if (E->sphere.caps == 12) {
            /* VTK multiblock file */
            write_vtm(E, cycles);
            /* LLNL VisIt multiblock file */
            write_visit(E, cycles);
        }
        else
            write_pvts(E, cycles);
    }
}
