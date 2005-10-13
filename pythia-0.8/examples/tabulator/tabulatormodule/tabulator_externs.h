// -*- C++ -*-
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//
//                             Michael A.G. Aivazis
//                      California Institute of Technology
//                      (C) 1998-2005  All Rights Reserved
//
// <LicenseText>
//
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//

#if !defined(tabulator_h)
#define tabulator_h

#if defined(NEEDS_F77_TRANSLATION)

#if defined(F77EXTERNS_LOWERCASE_TRAILINGBAR)

#define quadratic_f quadratic_
#define exponential_f exponential_
#define quadratic_set_f quadratic_set_
#define exponential_set_f exponential_set_

#define simpletab_f simpletab_
#define tabulator_f tabulator_


#elif defined(F77EXTERNS_NOTRAILINGBAR)

#define quadratic_f quadratic
#define exponential_f exponential
#define quadratic_set_f quadratic_set
#define exponential_set_f exponential_set

#define simpletab_f simpletab
#define tabulator_f tabulator

#elif defined(F77EXTERNS_EXTRATRAILINGBAR)

#define quadratic_f quadratic__
#define exponential_f exponential__
#define quadratic_set_f quadratic_set__
#define exponential_set_f exponential_set__

#define simpletab_f simpletab__
#define tabulator_f tabulator__

#elif defined(F77EXTERNS_UPPERCASE_NOTRAILINGBAR)

#define quadratic_f QUADRATIC
#define exponential_f EXPONENTIAL
#define quadratic_set_f QUADRATIC_SET
#define exponential_set_f EXPONENTIAL_SET

#define simpletab_f SIMPLETAB
#define tabulator_f TABULATOR

#elif defined(F77EXTERNS_COMPAQ_F90)

// symbols that contain underbars get two underbars at the end
// symbols that do not contain underbars get one underbar at the end
// this applies to the FORTRAN external, not the local macro alias!!!

#define quadratic_f quadratic_
#define exponential_f exponential_
#define quadratic_set_f quadratic_set__
#define exponential_set_f exponential_set__

#define simpletab_f simpletab_
#define tabulator_f tabulator_

#else
#error Unknown translation for FORTRAN external symbols
#endif

#endif

extern "C" {

    void exponential_set_f(double *);
    void quadratic_set_f(double *, double *, double *);

    double quadratic_f(double *);
    double exponential_f(double *);

    void simpletab_f(double *, double *, double *, double *);

    typedef double (*model_t)(double *);
    void tabulator_f(double *, double *, double *, model_t);
}

#endif

// version
// $Id: tabulator_externs.h,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

// Generated automatically by CxxMill on Thu Apr 17 10:21:51 2003

// End of file 
