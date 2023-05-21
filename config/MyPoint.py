class Point(object):
    def __init__(self,x,y):
        self.x=x
        self.y=y

def compare(point1,pt):
    for i in range(len(pt)):
        if(pt[i].x==point1.x and pt[i].y==point1.y):
            return  True
