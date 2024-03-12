from ucimlrepo import fetch_ucirepo 

from elastic_clustering import elmap_class

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
  
# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (as pandas dataframes) 
X = np.array(wine_quality.data.features)
y = np.array(wine_quality.data.targets )
  
# metadata 
print(wine_quality.metadata) 
  
# variable information 
print(wine_quality.variables) 



print(X.shape)
print(y.shape)

#elmap = elmap_class(lmbda=1.0)
#elmap.fit(X)
#elmap.display_info()
#cluster_list = elmap.cluster_list
#np.savetxt('wine_cluster_results.txt', elmap.cluster_list)

cluster_list = np.loadtxt('wine_cluster_results.txt')

var_features = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

target_var = 'quality'

#fig = plt.figure(figsize=(16, 12))
#ax = plt.subplot(3, 4, 1)
#colors = list(mcolors.TABLEAU_COLORS.keys())
#colors.append('m')
#print(len(colors))
#
#for i in range(len(var_features)):
#    ax = plt.subplot(3, 4, i+1)
#    ax.set_title(var_features[i])
#    for j in range(len(cluster_list)):
#        ax.scatter(X[j, i], y[j], color=colors[int(cluster_list[j])], marker='.', s=4)
##plt.show()
#fig.savefig('wine_quality_plot.png', dpi=300)
#plt.close('all')


fig = plt.figure(figsize=(16, 12))
ax = plt.subplot(1, 1, 1)
colors = list(mcolors.TABLEAU_COLORS.keys())
colors.append('m')
print(len(colors))

bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

cluster_list_list = [[] for _ in range( int((max(cluster_list) + 1)))]
print(cluster_list_list)

#for i in range(int(max(cluster_list) + 1)):
#    ax = plt.subplot(3, 4, i+1)
#    ax.set_title('Cluster ' + str(i+1))
#    for j in range(len(cluster_list)):
#        if cluster_list[j] == i:
#            cluster_list_list[i].append(y[j][0])
#    print(cluster_list_list[i])
#    ax.hist(np.array(cluster_list_list[i]), bins)
#ax = plt.subplot(3, 4, 12)
ax.set_title('All')
ax.hist(y, bins)
plt.show()