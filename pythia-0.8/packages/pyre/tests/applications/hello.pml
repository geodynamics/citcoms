<?xml version="1.0"?>
<!--
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!
!                             Michael A.G. Aivazis
!                      California Institute of Technology
!                      (C) 1998-2005  All Rights Reserved
!
! <LicenseText>
!
! ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
-->

<!DOCTYPE inventory>

<inventory>

    <component name="hello">
        <property name="name">Michael Aivazis</property>
        <property name="address">1200 East California Blvd, Pasadenae, CA 91125</property>

        <property name="help">no</property>
        <property name="help-properties">no</property>
        <property name="help-components">no</property>

        <component name="journal">
            <property name="device">console</property>

            <component name="file">
                <property name="name">hello.log</property>
            </component>

            <component name="remote">
                <property name="key">deadbeef</property>
            </component>

            <component name="debug">
                <property name="hello">on</property>
            </component>
        </component>
    </component>

</inventory>

<!-- version-->
<!-- $Id: hello.pml,v 1.4 2005/03/10 21:36:12 aivazis Exp $-->

<!-- End of file -->
