import numpy as np  #import numpy
import matplotlib.pyplot as plt #import matplotlib
import random

def cluster_insert(X, mu): #define a cluster
    cluster = {}
  # read about lambdas and np.linalg.form
  # https://stackoverflow.com/questions/32141856/is-norm-equivalent-to-euclidean-distance ,
  # here we are using order 1 to calculate normalized distance
    for x in X:
        value = min([(i[0],np.linalg.norm(x - mu[i[0]]))for i in enumerate(mu)], key=lambda s:s[1])[0]
        try:
            cluster[value].append(x)
        except:
            cluster[value] = [x]
    return cluster


def new_center(mu,cluster):
    keys =sorted(cluster.keys())
    newmu = np.array([(np.mean(cluster[k],axis = 0))for k in keys])
    return newmu

def matched(newmu, oldmu):
    return (set([tuple(a)for a in newmu]) == set([tuple(a)for a in oldmu]))

def Apply_Kmeans(X, K, N):
    # selecting random centroids from dataset and by number of clusters.
    t1 = np.random.randint(N, size = K)
    oldmu = np.array([X[i]for i in t1])

    t2 = np.random.randint(N, size = K)
    newmu = np.array([X[i]for i in t2])

    cluster = cluster_insert(X, oldmu)
    itr = 0
    print("Graph after selecting initial clusters with initial centroids:")
    plot_cluster(oldmu,cluster,itr)
    while not matched(newmu, oldmu):
        itr = itr + 1
        oldmu = newmu
        cluster= cluster_insert(X,newmu)
        plot_cluster(newmu, cluster,itr)
        newmu = new_center(newmu,cluster)


    plot_cluster(newmu, cluster, itr)
    return

def plot_cluster(mu,cluster, itr):
    color = 10 * ['y.','r.','k.','c.','b.','m.'] # It is for different colors for markers
    print('Iteration number : ',itr)
    for l in cluster.keys():
        for m in range(len(cluster[l])):
            plt.plot(cluster[l][m][0], cluster[l][m][1], color[l], markersize=10)
    plt.scatter(mu[:,0],mu[:,1],marker = 'x', s = 150, linewidths = 5, zorder = 10)
    plt.show()

def init_graph(N, p1, p2):
    X = np.array([(random.choice(p1),random.choice(p2))for i in range(N)])  #random choice will pick a number from below list
    return X


def Simulate_Clusters():
    print(".........Starting Cluster Simulation.........")
    a = np.array([51.4,45.6,24.4,52.6,59.69,16.49,33.49,57.99,20.16,14.89,53,61,26,27,28,29,30,35,36,37,38,39,40])
    b = np.array([45,80,97,33,86,45,33,89,98,99,65,45,56,77])
    X = init_graph(len(a),b,a)
    plt.scatter(X[:, 0], X[:, 1]) #it will show the graph of array list
    plt.show()
    temp = Apply_Kmeans(X, len(b), len(X))


if __name__ == '__main__':
    Simulate_Clusters()