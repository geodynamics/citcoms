#ifdef USE_PETSC

#include "global_defs.h"
#include "element_definitions.h"
#include "petsc_citcoms.h"


/* return ||V||^2 */
double global_v_norm2_PETSc( struct All_variables *E,  Vec v )
{
    int i, m, d;
    int eqn1, eqn2, eqn3;
    double prod = 0.0, temp = 0.0;
    PetscErrorCode ierr;

    PetscScalar *V;
    ierr = VecGetArray( v, &V ); CHKERRQ( ierr );
    for (m=1; m<=E->sphere.caps_per_proc; m++)
        for (i=1; i<=E->lmesh.nno; i++) {
            eqn1 = E->id[m][i].doff[1];
            eqn2 = E->id[m][i].doff[2];
            eqn3 = E->id[m][i].doff[3];
            /* L2 norm  */
            temp += (V[eqn1] * V[eqn1] +
                     V[eqn2] * V[eqn2] +
                     V[eqn3] * V[eqn3]) * E->NMass[m][i];
        }
    ierr = VecRestoreArray( v, &V ); CHKERRQ( ierr );

    MPI_Allreduce(&temp, &prod, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return (prod/E->mesh.volume);
}


/* return ||P||^2 */
double global_p_norm2_PETSc( struct All_variables *E,  Vec p )
{
    int i, m;
    double prod = 0.0, temp = 0.0;
    PetscErrorCode ierr;

    PetscScalar *P;
    ierr = VecGetArray( p, &P ); CHKERRQ( ierr );

    for (m=1; m<=E->sphere.caps_per_proc; m++)
        for (i=0; i<E->lmesh.npno; i++) {
            /* L2 norm */
            temp += P[i] * P[i] * E->eco[m][i+1].area;
        }
    ierr = VecRestoreArray( p, &P ); CHKERRQ( ierr );

    MPI_Allreduce(&temp, &prod, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return (prod/E->mesh.volume);
}

/* return ||A||^2, where A_i is \int{div(u) d\Omega_i} */
double global_div_norm2_PETSc( struct All_variables *E,  Vec a )
{
    int i, m;
    double prod = 0.0, temp = 0.0;
    PetscErrorCode ierr;

    PetscScalar *A;
    ierr = VecGetArray( a, &A ); CHKERRQ( ierr );


    for (m=1; m<=E->sphere.caps_per_proc; m++)
        for (i=0; i<E->lmesh.npno; i++) {
            /* L2 norm of div(u) */
            temp += A[i] * A[i] / E->eco[m][i+1].area;

            /* L1 norm */
            /*temp += fabs(A[i]);*/
        }
    ierr = VecRestoreArray( a, &A ); CHKERRQ( ierr );

    MPI_Allreduce(&temp, &prod, 1, MPI_DOUBLE, MPI_SUM, E->parallel.world);

    return (prod/E->mesh.volume);
}

/* =====================================================
   Assemble grad(rho_ref*ez)*V element by element.
   Note that the storage is not zero'd before assembling.
   =====================================================  */

PetscErrorCode assemble_c_u_PETSc( struct All_variables *E, Vec U, Vec result, int level )
//                  double **U, double **result, int level)
{
    int e,j1,j2,j3,p,a,b,m;

    const int nel = E->lmesh.NEL[level];
    const int ends = enodes[E->mesh.nsd];
    const int dims = E->mesh.nsd;
    const int npno = E->lmesh.NPNO[level];

    PetscErrorCode ierr;
    PetscScalar *U_temp, *result_temp;

    ierr = VecGetArray( U, &U_temp ); CHKERRQ( ierr );
    ierr = VecGetArray( result, &result_temp ); CHKERRQ( ierr );

  for( m = 1; m <= E->sphere.caps_per_proc; m++ ) {
    for(a=1;a<=ends;a++) {
      p = (a-1)*dims;
      for(e=0;e<nel;e++) {
        b = E->IEN[level][m][e+1].node[a];
        j1= E->ID[level][m][b].doff[1];
        j2= E->ID[level][m][b].doff[2];
        j3= E->ID[level][m][b].doff[3];

        result_temp[e]  += E->elt_c[level][m][e+1].c[p  ][0] * U_temp[j1]
                         + E->elt_c[level][m][e+1].c[p+1][0] * U_temp[j2]
                         + E->elt_c[level][m][e+1].c[p+2][0] * U_temp[j3];
      }
    }
  }
  ierr = VecRestoreArray( U, &U_temp ); CHKERRQ( ierr );
  ierr = VecRestoreArray( result, &result_temp ); CHKERRQ( ierr );

  PetscFunctionReturn(0);
}

void strip_bcs_from_residual_PETSc( 
    struct All_variables *E, Vec Res, int level )
{
  int i, m, low, high;
  PetscErrorCode ierr;
  PetscScalar *ResData;
  ierr = VecGetArray(Res, &ResData);
    if( E->num_zero_resid[level][1] ) {
      for( i = 1; i <= E->num_zero_resid[level][1]; i++ ) {
        ResData[E->zero_resid[level][1][i]] = 0.0;
      }
    }
  ierr = VecRestoreArray(Res, &ResData);
}

PetscErrorCode initial_vel_residual_PETSc( struct All_variables *E,
                                 Vec V, Vec P, Vec F,
                                 double acc )
{
    int neq = E->lmesh.neq;
    int lev = E->mesh.levmax;
    int npnp = E->lmesh.npno;
    int i, m, valid;
    PetscErrorCode ierr;

    Vec u1;
    ierr = VecCreateMPI( PETSC_COMM_WORLD, neq+1, PETSC_DECIDE, &u1 ); 
    CHKERRQ( ierr );

    /* F = F - grad(P) - K*V */
    // u1 = grad(P) i.e. G*P
    ierr = MatMult( E->G, P, u1 ); CHKERRQ( ierr );
    // F = F - u1
    ierr = VecAXPY( F, -1.0, u1 ); CHKERRQ( ierr ); 
    // u1 = del2(V) i.e. K*V
    ierr = MatMult( E->K, V, u1 ); CHKERRQ( ierr );
    // F = F - u1
    ierr = VecAXPY( F, -1.0, u1 ); CHKERRQ( ierr ); 

    strip_bcs_from_residual_PETSc(E, F, lev);

    /* solve K*u1 = F for u1 */
    //ierr = KSPSetTolerances( ... );
    ierr = KSPSolve( E->ksp, F, u1 ); CHKERRQ( ierr );

    strip_bcs_from_residual_PETSc(E, u1, lev);

    /* V = V + u1 */
    ierr = VecAXPY( V, 1.0, u1 ); CHKERRQ( ierr );
  PetscFunctionReturn(0);
}

PetscErrorCode PC_Apply_MultiGrid( PC pc, Vec x, Vec y )
{

  PetscErrorCode ierr;
  struct MultiGrid_PC *ctx;
  PetscScalar *xData, *yData;
  ierr = PCShellGetContext( pc, (void **)&ctx ); CHKERRQ( ierr );
  int count, valid;
  double residual;
  int m, i;

  ierr = VecGetArray(x, &xData); CHKERRQ(ierr);
  for(i = 0; i < ctx->nno; ++i)
    ctx->RR[1][i] = xData[i];
  ierr = VecRestoreArray(x, &xData); CHKERRQ(ierr);
  /* initialize the space for the solution */
  for( i = 0; i < ctx->nno; i++ )
      ctx->V[1][i] = 0.0;

  count = 0;

  do {
    residual = multi_grid( ctx->E, ctx->V, ctx->RR, ctx->acc, ctx->level );
    valid = (residual < ctx->acc) ? 1 : 0;
    count++;
  } while ( (!valid) && (count < ctx->max_vel_iterations) );
  ctx->status = residual;
  
  ierr = VecGetArray(y, &yData); CHKERRQ(ierr);
  for(i = 0; i < ctx->nno; i++)
    yData[i] = ctx->V[1][i];
  ierr = VecRestoreArray(y, &yData); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode MatShellMult_del2_u( Mat K, Vec U, Vec KU )
{
  // K is neq x neq
  // U is neq x 1
  // KU is neq x 1
  int i, j, neq;
  PetscErrorCode ierr;
  PetscScalar *UData, *KUData;
  struct MatMultShell *ctx;
  MatShellGetContext( K, (void **)&ctx );
  neq = ctx->iSize; // ctx->iSize SHOULD be the same as ctx->oSize
  ierr = VecGetArray(U, &UData); CHKERRQ(ierr);
  for(j = 0; j <neq; j++)
    ctx->iData[1][j] = UData[j];
  ierr = VecRestoreArray(U, &UData); CHKERRQ(ierr);
  // actual CitcomS operation
  assemble_del2_u( ctx->E, ctx->iData, ctx->oData, ctx->level, 1 );
  ierr = VecGetArray(KU, &KUData); CHKERRQ(ierr);
  for(j = 0; j < neq; j++)
    KUData[j] = ctx->oData[1][j];
  ierr = VecRestoreArray(KU, &KUData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShellMult_grad_p( Mat G, Vec P, Vec GP )
{
  // G is neq x nel
  // P is nel x 1
  // GP is neq x 1
  int i, j, neq, nel;
  PetscErrorCode ierr;
  PetscScalar *PData, *GPData;
  struct MatMultShell *ctx;
  MatShellGetContext( G, (void **)&ctx );
  nel = ctx->iSize;
  neq = ctx->oSize;
  ierr = VecGetArray(P, &PData); CHKERRQ(ierr);
  for(j = 0; j < nel; j++)
    ctx->iData[1][j] = PData[j];
  ierr = VecRestoreArray(P, &PData); CHKERRQ(ierr);
  // actual CitcomS operation
  assemble_grad_p( ctx->E, ctx->iData, ctx->oData, ctx->level );
  ierr = VecGetArray(GP, &GPData); CHKERRQ(ierr);
  for(j = 0; j < neq; j++)
    GPData[j] = ctx->oData[1][j];
  ierr = VecRestoreArray(GP, &GPData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShellMult_div_u( Mat D, Vec U, Vec DU )
{
  // D is nel x neq
  // U is neq x 1
  // DU is nel x 1
  int i, j, neq, nel;
  PetscErrorCode ierr;
  PetscScalar *UData, *DUData;
  struct MatMultShell *ctx;
  MatShellGetContext( D, (void **)&ctx );
  neq = ctx->iSize;
  nel = ctx->oSize;
  ierr = VecGetArray(U, &UData); CHKERRQ(ierr);
  for(j = 0; j < neq; j++)
    ctx->iData[1][j] = UData[j];
  ierr = VecRestoreArray(U, &UData); CHKERRQ(ierr);
  // actual CitcomS operation
  assemble_div_u( ctx->E, ctx->iData, ctx->oData, ctx->level );
  ierr = VecGetArray(DU, &DUData); CHKERRQ(ierr);
  for(j = 0; j < nel; j++)
    DUData[j] = ctx->oData[1][j];
  ierr = VecRestoreArray(DU, &DUData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShellMult_div_rho_u( Mat DC, Vec U, Vec DU )
{
  // DC is nel x neq
  // U is neq x 1
  // DU is nel x 1
  int i, j, neq, nel;
  PetscErrorCode ierr;
  PetscScalar *UData, *DUData;
  struct MatMultShell *ctx;
  MatShellGetContext( DC, (void **)&ctx );
  neq = ctx->iSize;
  nel = ctx->oSize;
  ierr = VecGetArray(U, &UData); CHKERRQ(ierr);
  for(j = 0; j < neq; j++)
    ctx->iData[1][j] = UData[j];
  ierr = VecRestoreArray(U, &UData); CHKERRQ(ierr);
  // actual CitcomS operation
  assemble_div_rho_u( ctx->E, ctx->iData, ctx->oData, ctx->level );
  ierr = VecGetArray(DU, &DUData); CHKERRQ(ierr);
  for(j = 0; j < nel; j++)
    DUData[j] = ctx->oData[1][j];
  ierr = VecRestoreArray(DU, &DUData); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* USE_PETSC */
