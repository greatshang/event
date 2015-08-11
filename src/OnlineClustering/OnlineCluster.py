#!/usr/bin/env python 

# You may have this code under the 
# Do What The Fuck You Want To Public License 
# http://sam.zoy.org/wtfpl/

# Some more info is at:
# http://gromgull.net/blog/2009/08/online-clustering-in-python/

import sys
import math
import heapq
import operator
import random
import codecs
import scipy
import numpy


def kernel_linear(x,y):
    return scipy.dot(x,y)

def kernel_poly(x,y,a=1.0,b=1.0,p=2.0):
    return (a*scipy.dot(x,y)+b)**p

def kernel_gauss(x,y, sigma=0.00001):
    v=x-y
    l=math.sqrt(scipy.square(v).sum())
    return math.exp(-sigma*(l**2))

def kernel_normalise(k): 
    return lambda x,y: k(x,y)/math.sqrt(k(x,x)+k(y,y))

kernel=kernel_normalise(kernel_gauss)

def kernel_dist(x,y):
    # if gaussian kernel:
    return 2-2*kernel(x,y)
    # if not
    #return kernel(x,x)-2*kernel(x,y)+kernel(y,y)
def cos_dist(x,y):
    return numpy.linalg.norm(x-y)

def cos_sim(a,b):
    return numpy.dot(a,b)/(numpy.linalg.norm(a) * numpy.linalg.norm(b))
class Cluster(object):
    def __init__(self, a): 
        self.center=a
        self.size=0
    def add(self, e):
        self.size+=kernel(self.center, e)
        self.center+=(e-self.center)/self.size
    def merge(self, c):        
        self.center=(self.center*self.size+c.center*c.size)/(self.size+c.size)
        self.size+=c.size
    def resize(self,dim):
        extra=scipy.zeros(dim-len(self.center))
        self.center=scipy.append(self.center, extra)
    def __str__(self):
        return "Cluster( %s, %f )"%(self.center, self.size)

class Cluster4Cos(object):
    def __init__(self, a = None): 
        self.center=a
        self.size=int(1)
    def reset(self, center,size):
        self.center = center
        self.size = size 
    def add(self, e):
        self.center = (self.center*self.size+e)/(self.size+1)
        self.size+=1
    def resize(self,dim):
        extra=numpy.zeros(dim-len(self.center))
        self.center=numpy.append(self.center, extra)
    def __str__(self):
        return "Cluster( %s, %i )"%(self.center, self.size)
    
class Dist(object):
    """this is just a tuple,
    but we need an object so we can define lt for <"""
    def __init__(self,x,y,d):
        self.x=x
        self.y=y
        self.d=d
    def __lt__(self, obj):
        return self.d < obj.d
    def __str__(self):
        return "Dist(%f)"%(self.d)

class OnlineCluster(object) : 
    
    def __init__(self, N,threshold = 0.98):
        """N-1 is the largest number of clusters I can find
        Higher N makes me slower"""
        
        self.n=0
        self.N=N
        self.threshold = threshold
        self.clusters=[]
        # max number of dimensions we've seen so far
        self.dim=0 
        # cache inter-cluster distances
        self.dist=[]
    
    def loadClusters(self,file):
        self.clusters = []
        with codecs.open(file, 'r','utf-8') as f:
            for line in f:
                items = line.strip().split(' ')
                cluster = Cluster4Cos()
                cluster.reset(numpy.array([float(item) for item in items[1:]]), int(items[0]))
                self.clusters.append(cluster)
        
    def saveClusters(self,file):
        with codecs.open(file, 'w','utf-8') as f:
            for cluster in self.clusters:
                f.write(str(cluster.size)+' '+' '.join([str(number) for number in cluster.center]))
                f.write('\n')
    def resize(self, dim):
        for c in self.clusters:
            c.resize(dim)
        self.dim=dim
    def addNewCluster(self,e):
        newc=Cluster(e)
        self.clusters.append(newc)
        self.updatedist(newc)
    def onlineCluster(self, e):
        if len(e)>self.dim:
            self.resize(len(e))
        if len(self.clusters)>0: 
            # compare new points to each existing cluster
            print(e)
            c=[ ( i, cos_sim(x.center, e) ) for i,x in enumerate(self.clusters)]
            index,closetSim = max( c , key=operator.itemgetter(1))
            print(index,closetSim)
            if closetSim<self.threshold: # far from all clusters , should be a new cluster
                print('add a new cluster')
                newc=Cluster(e)
                self.clusters.append(newc)
            else:   # not far enough,should be attached to closest
                closest=self.clusters[index]
                closest.add(e)
        else:# make a new cluster for this point
            newc=Cluster(e)
            self.clusters.append(newc)
        self.n+=1       
    def cluster(self, e):
        if len(e)>self.dim:
            self.resize(len(e))
        print(len(self.clusters))
        if len(self.clusters)>0: 
            # compare new points to each existing cluster
            c=[ ( i, kernel_dist(x.center, e) ) for i,x in enumerate(self.clusters)]
            print(c)
            index,closetDist = min( c , key=operator.itemgetter(1))
            closest=self.clusters[index]
            print(closetDist)
            print(closest)
            print()
            closest.add(e)
            # invalidate dist-cache for this cluster
            self.updatedist(closest)
        if len(self.clusters)>=self.N and len(self.clusters)>1:
            # merge closest two clusters
            m=heapq.heappop(self.dist)
            m.x.merge(m.y)
            self.clusters.remove(m.y)
            self.removedist(m.y)
            self.updatedist(m.x)
        # make a new cluster for this point
        self.addNewCluster(e)
        self.n+=1
    
    def removedist(self,c):
        """invalidate intercluster distance cache for c"""
        r=[]
        for x in self.dist:
            if x.x==c or x.y==c: 
                r.append(x)
        for x in r: self.dist.remove(x)
        heapq.heapify(self.dist)
    def updatedist(self, c):
        """Cluster c has changed, re-compute all intercluster distances"""
        self.removedist(c)

        for x in self.clusters:
            if x==c: continue
            d=kernel_dist(x.center,c.center)
            t=Dist(x,c,d)
            heapq.heappush(self.dist,t)
                
    def trimclusters(self):
        """Return only clusters over threshold"""
        t=scipy.mean([x.size for x in filter(lambda x: x.size>0, self.clusters)])*0.1
        return filter(lambda x: x.size>=t, self.clusters)
    

def testFuc():
    import random
    import time
    import pylab
    plot=True
    
    points=[]
    # create three random 2D gaussian clusters
    for i in range(8):
        x=random.random()*3
        y=random.random()*3
        c=[scipy.array((x+random.normalvariate(0,0.1), y+random.normalvariate(0,0.1))) for j in range(100)]
        points+=c

    if plot: pylab.scatter([x[0] for x in points], [x[1] for x in points])

    random.shuffle(points)
    n=len(points)
    start=time.time()
    # the value of N is generally quite forgiving, i.e.
    # giving 6 will still only find the 3 clusters.
    # around 10 it will start finding more
    c=OnlineCluster(8)
    while len(points)>0: 
        c.onlineCluster(points.pop())
    clusters=c.clusters
    #print ("I clustered %d points in %.2f seconds and found %d clusters."%(n, time.time()-start, len(clusters)))
    if plot: 
        cx=[x.center[0] for x in clusters]
        cy=[y.center[1] for y in clusters]
    
        pylab.plot(cx,cy,"ro")
        pylab.draw()
        pylab.show() 
if __name__=="__main__": 
    testFuc()
#     test = Cluster4Cos(numpy.array([0.1,0.2,0.3]))
#     print(test.center)
#     print(test.size)
#     line = str(test.size)+' '+' '.join([str(number) for number in test.center])
#     #print(test)
#     items = line.strip().split(' ')
#     print(items)
#     size = int(items[0])
#     arrays = numpy.array([float(item) for item in items[1:]])
#     print(size)
#     print(arrays)
    

        