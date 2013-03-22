import numpy as np
import matplotlib.pyplot as plt

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


class KDTree:


    def __init__(self):
        self.depth = 0 
        self.node = KDTreeNode()
       
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

    def 

def draw(node, minx, maxx, miny, maxy):
    """Draw a kd-tree been built, points must have two dimensions.

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
  
# test       
points = np.array([(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)])
kdtree = KDTree() 
kdtree.build(points)

plt.xlim(0, 10)
plt.ylim(0, 10)
draw(kdtree.node, 0, 10, 0, 10)
plt.show()

