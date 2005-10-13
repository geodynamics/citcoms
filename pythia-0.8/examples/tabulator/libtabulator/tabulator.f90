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

subroutine tabulator(low, hi, step, func)

    implicit none

    double precision:: low, hi, step
    double precision:: x, r_func

    double precision:: func
    external func

    print '(1x,A15,A3,A15)', 'x       ', ' | ', 'value     '
    print '(1x,A33)', '----------------+----------------'

    do x = low, hi, step
        r_func = func(x)
        print '(1x,F15.9,A3,F15.9)', x, ' | ', r_func
    end do

end subroutine tabulator


! version
! $Id: tabulator.f90,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

! End of file 
