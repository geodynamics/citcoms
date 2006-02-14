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

module exponential_parameters
    double precision:: a
end module exponential_parameters

!
subroutine exponential_set(new_a)
    use exponential_parameters

    implicit none
    double precision:: new_a

    a = new_a

    return
end subroutine exponential_set

!
function exponential(x)
    use exponential_parameters

    implicit none

    double precision:: exponential, x

    exponential = dexp(a*x)

end function exponential

! version
! $Id: exponential.f90,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

! End of file 
