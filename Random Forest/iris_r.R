ind <- sample(2,nrow(iris),replace=TRUE,prob=c(0.7,0.3))
trainData <- iris[ind==1,]
testData <- iris[ind==2,]
#Load Library Random Forest
library(randomForest)

#Generate Random Forest learning tree
iris_rf <- randomForest(Species~.,data=trainData,ntree=100,proximity=TRUE)
table(predict(iris_rf),trainData$Species)

print(iris_rf)

plot(iris_rf)

importance(iris_rf)

varImpPlot(iris_rf)


#Try to build random forest for testing data
irisPred<-predict(iris_rf,newdata=testData)
table(irisPred, testData$Species)

#Try to see the margin, positive or negative, if positif it means correct classification
plot(margin(iris_rf,testData$Species))

#Try to tune Random Forest
tune.rf <- tuneRF(iris[,-5],iris[,5], stepFactor=0.5)

print(tune.rf)
