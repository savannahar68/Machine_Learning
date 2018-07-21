In this part we'll learn classification, which is basically grouping of data points.<br/>
Whenever we have scattered data points into clusters(groups) we can recognize it using naked eye <br/>
Similarly we can use Machine learning to determine to which cluster the new point should go.

1.K Nearest Neighbours - 
	In K Nearest Neighbours we ususally take k as odd number(To solve the conflict which take place when we get same cluster number for a point) and then find K closest point to the new point , now according to majority we put the point in that particular cluster.For finding the distance between the points we calculate the Euclidean Distance.
	The problem with the Euclidean distance is if we have very large dataset then the process is very exhaustive as it will take a lot of time, better solution to this approach is using a SVM.