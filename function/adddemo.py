import math

from args import args
from config.MyPoint import Point
from function.start import save_img
from config.MyPoint import compare


#注释的是方差
def add(iter,pt_arr,im1,im2,la,im22,sort,pt):
    pt_temp=[]
    for i in range(len(pt_arr)):
        point=pt_arr[i]
        t1=getarr(im22,point)
        for j in range(point.x-8,point.x+8,15):
            for k in range(point.y-8,point.y+8,15):
                if(j>=8 and k>=8 and j!=point.x and k!=point.y and j<im22.shape[0] and k<im22.shape[1]):
                   if(compare(Point(j,k),pt)):
                       continue
                   t2=getarr(im22,Point(j,k))
                   if(abs(t1-t2)<=args.fc):
                       pt_temp.append(Point(j,k))
                       pt.append(Point(j,k))
                       save_img(im1,im2,la,iter,j,k,sort)
    return  pt_temp
def getarr(im22,point1):
    sum1=0
    for i in range(point1.x-8,point1.x+8):
        for j in range(point1.y-8,point1.y+8):
               if(i>=0 and j>=0 and i<im22.shape[0] and j<im22.shape[1]):
                 sum1+=im22[i][j]
    avg=sum1/256
    t=0
    for i in range(point1.x-8,point1.x+8):
        for j in range(point1.y-8,point1.y+8):
            if (i >= 0 and j >= 0 and i<im22.shape[0] and j<im22.shape[1]):
               t+=(avg-im22[i][j])*(avg-im22[i][j])/256
    return math.sqrt(t)
