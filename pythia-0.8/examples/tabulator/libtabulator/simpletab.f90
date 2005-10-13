! -*- F90 -*-
!
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
!                             Michael A.G. Aivazis
!                      California Institute of Technology
!                      (C) 1998-2005  All Rights Reserved
!
! <LicenseText>
!
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!

subroutine simpletab(a, low, hi, step)

    implicit none
    double precision:: a, low, hi, step

    external exponential
    double precision:: exponential

    double precision:: x, r_exp

    call exponential_set(a)

    print '(1x,A15,A3,A15)', 'x       ', ' | ', 'value     '
    print '(1x,A33)', '----------------+----------------'

    do x = low, hi, step
        r_exp = exponential(x)
        print '(1x,F15.9,A3,F15.9)', x, ' | ', r_exp
    end do

end subroutine simpletab


! version
! $Id: simpletab.f90,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

! End of file 
