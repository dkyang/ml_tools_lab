import numpy as np
import matplotlib.pyplot as plt


class Kmeans:


    def __init__(self):
        #stopping criterion for iteration
        self.threshold = 0.5
     
        self.centers = None
        self.n_samples_in_class = None

    def cluster(self, samples, k):
        n_samples = len(samples)
        cluster_label = np.zeros(n_samples, 'int8')

        min_val = np.amin(samples)
        max_val = np.amax(samples)
        self.centers = min_val + np.random.rand(k, 2) * (max_val-min_val)

        self.n_samples_in_cluster = np.zeros(k)

        # iteration
        is_continue = True
        while is_continue:
            self.n_samples_in_cluster = np.zeros(k)

            # assign points to clusters
            for i in xrange(n_samples):
                min_dist = 100000
                label = -1
                for c in xrange(k):
                    dist = self.__distance__(samples[i], self.centers[c])
                    if dist < min_dist:
                        min_dist = dist
                        label = c
                cluster_label[i] = label
                self.n_samples_in_cluster[label] += 1

            # recompute centers of each cluster
            centers_subtract = self.centers
            self.centers = np.zeros((k, 2))
            for s in xrange(n_samples):
                label = cluster_label[s]
                self.centers[label] += samples[s]
            for c in xrange(k):
                self.centers[c] /= self.n_samples_in_cluster[c] 

            # judge whether the iteration will continue
            centers_subtract = abs(self.centers - centers_subtract)
            is_continue = False
            for c in centers_subtract:
                if c[0] > self.threshold or c[1] > self.threshold:
                    is_continue = True

        return cluster_label

    # compute euclidean distance 
    def __distance__(self, sample_lhs, sample_rhs):
        dist = np.sum(np.power(sample_lhs-sample_rhs, 2))
        return dist


# test k-means cluster
n_samples = 300
samples = np.random.rand(n_samples, 2) * 10

# To Fix: error occurs when k is less than 3
k = 3
kmeans = Kmeans()
cluster_label = kmeans.cluster(samples, k)
n_samples_in_cluster = kmeans.n_samples_in_cluster

# draw points according to their labels
cur_label_ind = np.zeros(k, dtype='int8')
sample_list = [[None]*n_samples_in_cluster[n] for n in xrange(k)]

# store points according to their labels
# sample_list[k] stores all the points of kth class 
for s in xrange(n_samples):
    label = cluster_label[s]
    label_ind = cur_label_ind[label]
    cur_label_ind[label] += 1
    sample_list[label][label_ind] = samples[s]

for label in xrange(k):
    points = np.array(sample_list[label])
    if not points.any():
        print "This happens when current class hava no point."  
    else:
        color = np.random.rand(3)
        plt.scatter(points[:,0], points[:,1], c=color)

plt.show()