import numpy as np
import matplotlib.pyplot as plt

MAX_DISTANCE = 1000000

class KDTreeNode:
    """Node of a kd-tree
    """
    def __init__(self):
        self.points = np.array([])
        self.axis = 0
        self.median = 0
        self.depth = 0
        self.left = None
        self.right = None

class Neighbor:


    def __init__(self, distance=MAX_DISTANCE, node=None):
        self.distance = distance
        self.node = node


class NeighborHeap:
    """A max-heap stores k candidate nearest neighbors,
    the neighbors are sorted by their distance between
    querying point.  
    """
    def __init__(self):
        self.array = []
    

class KDTree:


    def __init__(self):
        self.depth = 0 
        #self.node = KDTreeNode()
        self.node = None
       
    def build(self, points, depth=0):
        """Build a kd-tree from a set of k-dimensional points.

        Parameters
        ----------
        points : K-dimensional points we want to build from.

        depth : Depth of this node.

        Return
        ----------
        node : Root node of current sub-kdtree. 
        """
        if len(points) == 0:
            return 

        # select the median of the points as the pivot 
        median = len(points) / 2 
        self.node = KDTreeNode()
        self.node.median = median
        self.node.depth = depth

        # sort points along choosed planes
        self.node.axis = depth % 2
        arg = np.argsort(points[:,self.node.axis])
        points = points[arg,:]
        self.node.points = points
        
        # recursive build kd-tree
        left_child = KDTree()
        right_child = KDTree()
        self.node.left = left_child.build(points[:median],depth+1)
        self.node.right = right_child.build(points[median+1:],depth+1)

        return self.node

    def nearest_neighbor_search(self, target):
        if self.node is None:
            #print "kdtree have not been built."
            raise Exception

        node = self.node
        prev = node
        search_path = []
        while node is not None:
            search_path.append(node)
            axis = node.axis
            # pivot position
            median = node.median
            prev = node
            if target[axis] <= node.points[median, axis]:
                node = node.left
            else:
                node = node.right
        
        #nearest = prev
        nearest = search_path.pop()
        nearest_dist = self.__distance__(nearest.points[nearest.median], target)
        
        node = prev
        # back track
        #while node is not None:
        while search_path:
            node = search_path.pop()

            print node.points[node.median]
            print nearest_dist

            if node.left is None and node.right is None:    
                # leaf node
                dist = self.__distance__(node.points[node.median], target)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = node   
            else:               
                axis = node.axis
                median = node.median
                if abs(node.points[median, axis] - target[axis]) < nearest_dist:
                    dist = self.__distance__(node.points[node.median], target)
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest = node

                    if target[axis] <= node.points[median, axis]:
                        node = node.right 
                    else:
                        node = node.left

                    while node is not None:
                        search_path.append(node)
                        axis = node.axis
                        # pivot position
                        median = node.median
                        if target[axis] <= node.points[median, axis]:
                            node = node.left
                        else:
                            node = node.right

        return nearest, nearest_dist       

    def k_nearest_neighbor_search(self, target, k):
        if self.node is None:
            raise Exception

        node = self.node
        prev = node
        search_path = []
        while node is not None:
            print node.points[node.median]
            search_path.append(node)
            axis = node.axis
            # choose median as pivot position
            median = node.median
            if target[axis] <= node.points[median, axis]:
                node = node.left
            else:
                node = node.right

        # back track
        k_nearest_neighbors = []
        closest_dist = MAX_DISTANCE        
        while search_path:
            node = search_path.pop()

            if node.left is None and node.right is None:    
                # leaf node
                dist = self.__distance__(node.points[node.median], target)
                if len(k_nearest_neighbors) < k:
                    neighbor = Neighbor(dist, node)
                    k_nearest_neighbors.append(neighbor)
                    if len(k_nearest_neighbors) == k:
                        pos = self.__find_max_neighbor__(k_nearest_neighbors)
                        closest_dist = k_nearest_neighbors[pos].distance
                elif dist < closest_dist:
                    pos = self.__find_max_neighbor__(k_nearest_neighbors)
                    k_nearest_neighbors.pop(pos)
                    neighbor = Neighbor(dist, node)                
                    k_nearest_neighbors.append(neighbor)
                    pos = self.__find_max_neighbor__(k_nearest_neighbors)
                    closest_dist = k_nearest_neighbors[pos].distance
            else:               
                axis = node.axis
                median = node.median
                if abs(node.points[median, axis] - target[axis]) < closest_dist:
                    dist = self.__distance__(node.points[node.median], target)
                    if len(k_nearest_neighbors) < k:
                        neighbor = Neighbor(dist, node)
                        k_nearest_neighbors.append(neighbor)

                        # if after this insertion number of neighbors equals k,
                        # then we must update closest distance
                        if len(k_nearest_neighbors) == k:
                            pos = self.__find_max_neighbor__(k_nearest_neighbors)
                            closest_dist = k_nearest_neighbors[pos].distance
                    elif dist < closest_dist:
                        pos = self.__find_max_neighbor__(k_nearest_neighbors)
                        k_nearest_neighbors.pop(pos)
                        neighbor = Neighbor(dist, node)                
                        k_nearest_neighbors.append(neighbor)
                        pos = self.__find_max_neighbor__(k_nearest_neighbors)
                        closest_dist = k_nearest_neighbors[pos].distance
                    
                    if target[axis] <= node.points[median, axis]:
                        node = node.right 
                    else:
                        node = node.left

                    """If the hypersphere crosses the plane, there could 
                    be nearer points on the other side of the plane, so the
                    algorithm must move down the other branch of the tree from
                    the current node looking for closer points, following the
                    same recursive process as the entire search[wiki]
                    """
                    while node is not None:
                        search_path.append(node)
                        axis = node.axis
                        # pivot position
                        median = node.median
                        if target[axis] <= node.points[median, axis]:
                            node = node.left
                        else:
                            node = node.right

        return k_nearest_neighbors       
 
    def __find_max_neighbor__(self, neighbors):	
        max_dist = -1
        l = len(neighbors)
        for i in xrange(l):
            if max_dist < neighbors[i].distance:
                max_dist = neighbors[i].distance
                pos = i
        return pos
            

    def __distance__(self, point1, point2):
        if len(point1) != 2:
            print "WTF!!!"
        dist = np.sum(np.power(point1-point2, 2)) 
        return dist           

def draw(node, minx, maxx, miny, maxy):
    """Draw a kd-tree been built, points can only have two dimensions.

    Parameters
    ----------
    node : class 'KDTreeNode', root node of a kd-tree.

    minx : Minimum value when drawing line in horizontal direction.

    maxx : Maximum value when drawing line in horizontal direction.

    miny : Minimum value when drawing line in vertical direction.

    maxy : Maximum value when drawing line in vertical direction.

    Returns
    -------
    None
    """
    if len(points[0]) != 2:
        print "The points in kd-tree must have two dimensions."
        return
 
    if node is None:
        return
    
    # draw pivot point in this node
    x = node.points[node.median, 0]
    y = node.points[node.median, 1]
    plt.scatter(x, y)

    # draw a vertical or horizontal line to show the splitting hyperplane 
    if node.axis == 0:
        yind = np.array(xrange(miny, maxy+1))
        plt.plot(x + 0*yind, yind, 'r')
        draw(node.left, minx, x, miny, maxy)
        draw(node.right, x, maxx, miny, maxy)
    elif node.axis == 1:
        xind = np.array(xrange(minx, maxx+1))
        plt.plot(xind, y + 0*xind, 'b')
        draw(node.left, minx, maxx, miny, y)
        draw(node.right, minx, maxx, y, maxy)     

  
# build a kdtree using points below       
points = np.array([(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)])
kdtree = KDTree() 
kdtree.build(points)

plt.xlim(0, 10)
plt.ylim(0, 10)
draw(kdtree.node, 0, 10, 0, 10)

# just test exception mechanism in python :) 
target = np.array([6,1.5])
try:
    nearest_node, nearest_dist = kdtree.nearest_neighbor_search(target)
except Exception:
    if kdtree.node is None:
        print "kdtree have not been built."
print "nearest distance to target is", nearest_dist 

# test k nearest neighbors search
plt.scatter(target[0], target[1], s=30, c='r', marker='D')
k_nearest_neighbors = kdtree.k_nearest_neighbor_search(target, 3)

x = []
y = []
for neighbor in k_nearest_neighbors:
    node = neighbor.node
    x.append(node.points[node.median, 0])
    y.append(node.points[node.median, 1])
plt.scatter(x, y, s=30, c='y', marker='D')
plt.show()