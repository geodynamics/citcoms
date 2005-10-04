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

#if !defined(CitcomS_parallel_related_h)
#define CitcomS_parallel_related_h

#ifdef __cplusplus
extern "C" {
#endif

void parallel_process_termination();
void parallel_process_sync(struct All_variables *E);
double CPU_time0();
void parallel_processor_setup(struct All_variables *E);
void parallel_domain_decomp0(struct All_variables *E);
void parallel_domain_boundary_nodes(struct All_variables *E);
void parallel_communication_routs_v(struct All_variables *E);
void parallel_communication_routs_s(struct All_variables *E);
void set_communication_sphereh(struct All_variables *E);
void exchange_id_d(struct All_variables *E, double **U, int lev);
void exchange_node_d(struct All_variables *E, double **U, int lev);
void exchange_node_f(struct All_variables *E, float **U, int lev);
void exchange_snode_f(struct All_variables *E, float **U1, float **U2, int lev);

#ifdef __cplusplus
}
#endif

#endif
