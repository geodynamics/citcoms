#!/usr/bin/env python

import sys, string, os
import math

#==============================================================
#==============================================================
def smooth_line(x,y,smoothing):

    xs=[]
    ys=[]
    m=len(x)-1
    print 'm=',m
    i=0
    while i<=m:
        #b, beginning of line; e, end of line
        Swxb=Swyb=Swxe=Swye=0.0
        Swb=Swe=0.0001
        dist=0.0
        j=0
        while j<i:
            k=i-j
            dist=dist+math.sqrt( (x[k]-x[i])**2 + (y[k]-y[i])**2 )
            wj=math.exp(-dist/smoothing)
            if i==k:
                wj=0.0
            Swxb=Swxb + wj*x[k] 
            Swyb=Swyb + wj*y[k] 
            Swb=Swb+wj
            j+=1
        dist=0.0
        j=i
        while j<=m:
            k=j
            dist=dist+math.sqrt( (x[k]-x[i])**2 + (y[k]-y[i])**2 )
            wj=math.exp(-dist/smoothing)
            if i==k:
                wj=0.0
            Swxe=Swxe + wj*x[k] 
            Swye=Swye + wj*y[k] 
            Swe=Swe+wj
            j+=1
        
        xvalue=(Swe*(Swxb/Swb)+Swb*(Swxe/Swe))/(Swb+Swe)
        yvalue=(Swe*(Swyb/Swb)+Swb*(Swye/Swe))/(Swb+Swe)
        if i<=1:
            xs.append(x[0])
            ys.append(y[0])
        elif i>1 and i<m:
            xs.append(xvalue)
            ys.append(yvalue)
        else:
            xs.append(x[m])
            ys.append(y[m])
        i+=1
    return xs,ys

#==============================================================

x=[]
y=[]
n=100

i=0
while i<=n:
   #xvalue=0.25+float(i)/float(n)
   if i <n/2:
       xvalue=0.25
       #yvalue=0.25+float(i)/float(n/2)
       yvalue=0.25+float(i)/float(n/2)
   else:
       xvalue=0.25+float(i-n/2)/float(n/2)
       #yvalue=0.25+1.0-float(i-n/2)/float(n/2)
       yvalue=1.25
   x.append(xvalue)
   y.append(yvalue)
   i+=1

#print x,y
s=0.10
xs,ys=smooth_line(x,y,s)
s=1.0
xs1,ys1=smooth_line(x,y,s)
s=5.00
xs2,ys2=smooth_line(x,y,s)

OL=open("original_line.xy","w")
SL=open("smooth_line.xy","w")
SL1=open("smooth_line_1.xy","w")
SL2=open("smooth_line_2.xy","w")
i=0
while i<=n:
    OL.write("%f  %f\n" % (x[i],y[i]))
    SL.write("%f  %f\n" % (xs[i],ys[i]))
    SL1.write("%f  %f\n" % (xs1[i],ys1[i]))
    SL2.write("%f  %f\n" % (xs2[i],ys2[i]))
    i+=1

OL.close()
SL.close()
SL1.close()
SL2.close()

psfile="plot.ps"
cmd="psxy original_line.xy -JX10.0 -R0.0/2.0/0.0/2.0 -W2.0 -P -X1.0 -Y1.0 -K  > %s" % (psfile)
print cmd
os.system(cmd)
cmd="psxy smooth_line.xy -JX10.0 -R0.0/2.0/0.0/2.0 -W4.0/255/0/0 -P -X0.0 -Y0.0 -O -K  >> %s" % (psfile)
print cmd
os.system(cmd)
cmd="psxy smooth_line_1.xy -JX10.0 -R0.0/2.0/0.0/2.0 -W4.0/0/255/0 -P -X0.0 -Y0.0 -O -K  >> %s" % (psfile)
print cmd
os.system(cmd)
cmd="psxy smooth_line_2.xy -JX10.0 -R0.0/2.0/0.0/2.0 -W4.0/0/0/255 -P -X0.0 -Y0.0 -O  >> %s" % (psfile)
print cmd
os.system(cmd)
