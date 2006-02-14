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

module quadratic_parameters
    double precision:: a, b, c
end module quadratic_parameters

!
subroutine quadratic_set(new_a, new_b, new_c)
    use quadratic_parameters

    implicit none
    double precision:: new_a, new_b, new_c

    a = new_a
    b = new_b
    c = new_c

    return
end subroutine quadratic_set

!
function quadratic(x)
    use quadratic_parameters

    implicit none
    double precision:: quadratic, x

    quadratic = a*x**2 + b*x + c

end function quadratic


! version
! $Id: quadratic.f90,v 1.1.1.1 2005/03/17 20:03:02 aivazis Exp $

! End of file 
