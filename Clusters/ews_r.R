library("dummies")
input <- read.csv("C:/Users/oyedeepak/Downloads/Assignment/Clustering/EastWestAirlines.csv", header = TRUE)

mydata <- input[1:3999,2:12] ## exclude the columns with university and state names
normalized_data <- scale(mydata) ## normalize the columns
set.seed(1112)

d <- dist(normalized_data, method = "euclidean") 
fit <- hclust(d, method="ward.D2")
plot(fit) # display dendrogram

plot(fit) # display dendrogram
rect.hclust(fit, k=3, border="red")

d <- dist(normalized_data, method = "euclidean") 
fit <- hclust(d, method="complete")
options(scipen=99999999)


groups <- cutree(fit, k=3)

clust.centroid = function(i, dat, groups) {
  ind = (groups == i)
  colMeans(dat[ind,])
}
sapply(unique(groups), clust.centroid, mydata, groups)


normalized_data_95 <- scale(mydata[sample(nrow(mydata), nrow(mydata)*.95), ])
d <- dist(normalized_data_95, method = "euclidean") 
fit <- hclust(d, method="ward.D2")
plot(fit)


fit <- kmeans(normalized_data, centers=3, iter.max=10)
fit$centers


## Determine number of clusters
Cluster_Variability <- matrix(nrow=8, ncol=1)
for (i in 1:8) Cluster_Variability[i] <- kmeans(normalized_data, centers=i)$tot.withinss
plot(1:8, Cluster_Variability, type="b", xlab="Number of clusters", ylab="Within groups sum of squares")


sapply(unique(groups), clust.centroid, mydata, groups)


fit <- kmeans(mydata, centers=3, iter.max=10)
t(fit$centers)

fit$size


#Total = 3118+81+800 = 3999
#C1 = 3118/3999 = 0.7796
#C1 = 81/3999 = 0.0202
#C1 = 800/3999 = 0.200
#Cluster1 has maximum number of travelers(= 78%), Cluster 2 has 20% of travelers.