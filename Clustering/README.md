So basically we are aiming at Unsupervised learning clustering, in which 2 types are there <br/>
So given just the Feature sets the machine finds for Groups or cluster's within them.


1.Flat Clustering - So we specify the machine the number of cluster we expect to find. <br/>
2.Hierarchical Clustering - In this the machine figures out the number of clusters <br/>

Algorithms - 
1.K-Means clustering - Comes under Flat type and we specify no. of clusters we want as parameter and it finds it <br/>
	At start we chose k random centroids and then calculate the distance(eucledian) of points which are near to it.After this we take mean of all points and then that becomes our new centroid.Again chose points which are near to it and repeat the same steps until you get the same positions.<br/>One disadvantage of this algo is it tries to make equal group of data, so sometimes that can be a problem.<br/>
2.Mean Shift - Comes under Hierarchical