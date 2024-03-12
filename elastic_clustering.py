import numpy as np
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as mcolors

import matplotlib as mpl
mpl.rc('font', family='Times New Roman', size=18)


def associate(data, nodes):
    n_pts, n_dims = np.shape(data)
    n_nodes, n_dims = np.shape(nodes)
    clusters = [-1 for _ in range(n_pts)]
    for i in range(n_pts):
        clusters[i] = np.argmin([np.linalg.norm(data[i] - nodes[j]) for j in range(n_nodes)])
    C = np.zeros((n_nodes, n_dims))
    for i in range(n_nodes):
        sum = C[i]
        for j in range(n_pts):
            if (clusters[j] == i):
                sum = sum + data[j]
        #C[i] = sum / n_pts
        C[i] = sum
    return clusters, C
    
def calc_A(clusters, E):
    A = np.copy(E)
    n_nodes, _ = np.shape(A)
    n_pts = len(clusters)
    C = np.array(clusters)
    for i in range(n_nodes):
        #A[i, i] = A[i, i] + (np.sum(C == i) / n_pts)
        A[i, i] = A[i, i] + np.sum(C == i)
    return A
    
#optimize elastic map, see "Principal Graphs and Manifolds" by Gorban & Zinovyev for details, slightly modified with negative stretching constant and no bending energy
def optimize_map(data, nodes, stretch=0.005):
    lmda = -stretch
    n_data, n_dims = np.shape(data)
    n_nodes, n_dims = np.shape(nodes)
    E = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            E[i, i] = E[i, i] + lmda
            E[j, j] = E[j, j] + lmda
            E[i, j] = E[i, j] - lmda
            E[j, i] = E[j, i] - lmda
            
    clusters, C = associate(data, nodes)
    A = calc_A(clusters, E)
    new_nodes = np.linalg.lstsq(A, C, rcond=None)[0]
    new_clusters, C = associate(data, new_nodes)
    iter = 1
    #print([iter, calc_nrg(data, new_nodes, new_clusters)])
    while (new_clusters != clusters) and (iter < 20):
        clusters = new_clusters
        nodes = new_nodes
        A = calc_A(clusters, E)
        new_nodes = np.linalg.lstsq(A, C, rcond=None)[0]
        new_clusters, C = associate(data, new_nodes)
        iter = iter + 1
        #print([iter, calc_nrg(data, new_nodes, new_clusters)])
    return new_nodes, new_clusters

def calc_nrg(data, nodes, clusters, lmda):
    n_pts, n_dims = np.shape(data)
    n_nodes, n_dims = np.shape(nodes)
    Uy = 0.
    for i in range(n_pts):
        Uy = Uy + np.linalg.norm(data[i] - nodes[clusters[i]])**2
    Ue = 0.
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            if (i != j):
                Ue = Ue + np.linalg.norm(nodes[i] - nodes[j])**2
    #return (Uy / n_pts) + (lmda * Ue)
    return Uy + (lmda * Ue)
    


class elmap_class(object):
    
    #set up variables
    def __init__(self, initial_points=[], nlabels=1, lmbda=0.005):
        self.colors = list(mcolors.TABLEAU_COLORS.keys())
        self.point_list = initial_points
        self.cluster_list = []
        self.cluster_centers = []
        self.num_points = len(self.point_list)
        self.num_labels = nlabels
        self.nrg = -1
        self.stretch = lmbda
        if self.num_points > 0:
            self.cluster_points()

    #get functions
    def get_nrg(self):
        return self.nrg

    def get_clusters(self):
        return self.cluster_list

    def get_centers(self):
        return self.cluster_centers
        
    def get_num_labels(self):
        return self.num_labels

    #cluster given points
    def cluster_points(self):
        if len(self.cluster_centers) == 0:
            #first time clustering, initialize map guess
            self.cluster_centers = self.point_list[np.random.choice(self.num_points, size=self.num_labels, replace=False)]
        
        #initial clustering
        self.cluster_centers, self.cluster_list = optimize_map(self.point_list, self.cluster_centers, stretch=self.stretch)
        self.nrg = calc_nrg(self.point_list, self.cluster_centers, self.cluster_list, lmda=self.stretch)
        
        #check if one more cluster reduces energy, keep going until finding local minimum
        check_next_cluster = True
        while (check_next_cluster):
            #check if adding another cluster reduces energy
            temp_cluster_centers = copy.deepcopy(self.cluster_centers)
            #temp_cluster_centers.append(self.point_list[-1])
            temp_cluster_centers = np.vstack((temp_cluster_centers, self.point_list[-1]))
            temp_cluster_centers, temp_cluster_list = optimize_map(self.point_list, temp_cluster_centers, stretch=self.stretch)
            temp_nrg = calc_nrg(self.point_list, temp_cluster_centers, temp_cluster_list, lmda=self.stretch)
            #print(self.nrg, len(self.cluster_centers), temp_nrg, len(temp_cluster_centers))
            if (temp_nrg < self.nrg):
                #better to add another cluster
                self.cluster_centers = temp_cluster_centers
                self.cluster_list = temp_cluster_list
                self.nrg = temp_nrg
            else:
                #stop checking to see if another cluster should be added
                check_next_cluster = False
        self.num_labels = len(self.cluster_centers)
    
    #add points to the current list of points and cluster again
    def add_points(self, points):
        if self.num_points == 0:
            self.point_list = points
        else:
            self.point_list = np.vstack((self.point_list, points))  
        self.num_points = len(self.point_list)          
        self.cluster_points()
        
    #fit as another name for add points for functionality with sklearn
    def fit(self, X):
        self.add_points(X)
    
    #predict function to work with sklearn functionality, can be used to check what points would go into current clusters
    def predict(self, X):
        labels = np.array([0] * len(X))
        for idx in range(len(X)):
            #predict label as closest cluster center
            labels[idx] = np.argmin(np.linalg.norm(self.cluster_centers - X[idx], axis=1))
        print(len(self.cluster_centers))
        return labels
    
    #display clusters
    def display_info(self):
        print('Points:')
        print(self.point_list)
        print('Clusters:')
        print(self.cluster_list)
        print('Centers:')
        print(self.cluster_centers)
        print('Number of Centers:')
        print(len(self.cluster_centers))
        
    #plot the clusters
    def plot(self, mode='show', title='', fpath=''):
        n_pts, n_dims = np.shape(self.point_list)
        #self.display_info()
        if n_dims == 2:
            fig = plt.figure()
            plt.axis('equal')
            plt.title(title)
            for i in range(self.num_labels):
                plt.plot(self.cluster_centers[i][0], self.cluster_centers[i][1], color=self.colors[i], marker='*', ms=10)
                #print(self.point_list[np.array(self.cluster_list) == i])
                plt.scatter(self.point_list[np.array(self.cluster_list) == i][:, 0], self.point_list[np.array(self.cluster_list) == i][:, 1], color=self.colors[i], marker='.', s=8)
        elif n_dims == 3:
            print('Not yet implemented!')
        else:
            print('Plotting not supported in this dimension!')
        if mode == 'show':
            plt.show()
        else:
            plt.savefig(fpath + '/' + title + '.png', dpi=300, bbox_inches='tight')
            plt.close('all')

#test with different numbers of points from different centers
def exp_points_main():
    PIC_FPATH = 'pictures/sim_data'
    np.random.seed(1)
    num_points_list = [10, 20, 30]
    n_dims = 2
    centers = [[0, 0], [1, 2], [2, -1]]
    std_dev = 0.5
    for n_pts in num_points_list:
        pts1 = np.random.normal(loc=centers[0], scale=std_dev, size=(n_pts, n_dims))
        pts2 = np.random.normal(loc=centers[1], scale=std_dev, size=(n_pts, n_dims))
        pts3 = np.random.normal(loc=centers[2], scale=std_dev, size=(n_pts, n_dims))
        pts = np.vstack((pts1, pts2, pts3))
        EMclass = elmap_class(pts, lmbda=0.4)
        EMclass.display_info()
        EMclass.plot(mode='save', title=str(n_pts) + ' Samples per Center', fpath=PIC_FPATH)

#test iterative abilities    
def exp_iterative_main():
    PIC_FPATH = 'pictures/sim_data'
    n_pts = 60
    n_dims = 2
    n_nodes = 1
    pts1 = np.random.normal(loc=[5, 5], size=(n_pts//3, n_dims))
    pts2 = np.random.normal(loc=[0, 0], size=(n_pts//3, n_dims))
    pts3 = np.random.normal(loc=[7, 0], size=(n_pts//3, n_dims))
    #pts = np.vstack((pts1, pts2, pts3))
    EMclass = elmap_class(pts1)
    EMclass.plot(mode='save', title='class20', fpath=PIC_FPATH)
    EMclass.add_points(pts2)
    EMclass.plot(mode='save', title='class40', fpath=PIC_FPATH)
    EMclass.add_points(pts3)
    EMclass.plot(mode='save', title='class60', fpath=PIC_FPATH)
    
#simple experiment to test functionality
def exp_simple_main():
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    n_pts = 20
    n_dims = 2
    n_nodes = 3
    pts = np.random.uniform(size=(n_pts, n_dims))
    nodes = pts[np.random.choice(n_pts, n_nodes, replace=False)]
    
    new_nodes, clusters = optimize_map(pts, nodes)
    
    plt.figure()
    for i in range(n_pts):
        plt.plot(pts[i, 0], pts[i, 1], colors[clusters[i]] + '.')
    plt.plot(new_nodes[:, 0], new_nodes[:, 1], 'ko')
    plt.show()
    

if __name__ == '__main__':
    exp_points_main()