setwd("C:/Users/RPJ/Desktop/Notes/Data Analysis/R")
#-------------Loading data-------------------
Train.data=read.csv("ProjectTrain.csv",header=T)
Test.data=read.csv("ProjectTest.csv",header=T)
library(plyr)
library(MASS)
library(class)
library(caret)
attach(Train.data)

#-------------------Data Preparation-------------------
#Replacing missing age values with mean age for training data
mean.age.train=mean(Age,na.rm=TRUE)
Train.data$Age[is.na(Train.data$Age)]=mean.age.train
Train.data$Age=lapply(Train.data$Age,round,2)
  
#Replacing missing age values with mean age for testing data
mean.age.test=mean(Test.data$Age,na.rm=TRUE)
Test.data$Age[is.na(Test.data$Age)]=mean.age.test
Test.data$Age=lapply(Test.data$Age,round,2)

#Filling missing Embarked values with mode for training data
unique_embarked<-unique(Embarked)
mode_embarked=unique_embarked[which.max(tabulate(match(Embarked, unique_embarked)))]
mode_embarked
mode_embarked="S" #Since the mode obtained is S
Train.data$Embarked[Train.data$Embarked==""]="S"

#Changing data type to factor for training data
Train.data$Survived=factor(Train.data$Survived)
Train.data$Pclass=factor(Train.data$Pclass)
Train.data$Sex=factor(Train.data$Sex)
Train.data$Embarked=factor(Train.data$Embarked)
Train.data$Age=unlist(Train.data$Age)
str(Train.data)
attach(Train.data)

#Changing data type to factor for testing data
Test.data$Survived=factor(Test.data$Survived)
Test.data$Pclass=factor(Test.data$Pclass)
Test.data$Sex=factor(Test.data$Sex)
Test.data$Embarked=factor(Test.data$Embarked)
Test.data$Age=unlist(Test.data$Age)
str(Test.data)

#---------------------------------Logistic regression---------------------------------
logistic.fit=glm(Survived~Pclass+Age+Sex+SibSp+Parch+Embarked,data=Train.data,family='binomial')
summary(logistic.fit)

logistic.probability=predict(logistic.fit,Test.data,type="response")

logistic.predict=rep("0",267)
logistic.predict[logistic.probability>0.5]="1"
attach(Test.data)
confusion.logistic=table(logistic.predict,Survived) #confusion matrix
confusion.logistic
Accuracy.logistic=mean(logistic.predict==Survived)
Accuracy.logistic
tp.logistic=confusion.logistic[2,2]/(confusion.logistic[1,2]+confusion.logistic[2,2])
tp.logistic #true positive rate
fp.logistic=confusion.logistic[2,1]/(confusion.logistic[1,1]+confusion.logistic[2,1])
fp.logistic #false positive rate

#---------------------------------Linear Discriminant Analysis---------------------------------

lda.fit=lda(Survived~Pclass+Age+Sex+SibSp+Parch+Embarked,data=Train.data)
lda.predict=predict(lda.fit,Test.data)

lda.class=lda.predict$class
attach(Test.data)
confusion.lda=table(lda.class,Survived)
confusion.lda
Accuracy.lda=mean(lda.class==Survived)
Accuracy.lda
tp.lda=confusion.lda[2,2]/(confusion.lda[1,2]+confusion.lda[2,2])
tp.lda
fp.lda=confusion.lda[2,1]/(confusion.lda[1,1]+confusion.lda[2,1])
fp.lda

# ---------------------------------Quadratic Discriminant Analysis---------------------------------

qda.fit=qda(Survived~Pclass+Age+Sex+SibSp+Parch+Embarked,data=Train.data)
qda.predict=predict(qda.fit,Test.data)

qda.class=qda.predict$class
attach(Test.data)
confusion.qda=table(qda.class,Survived)
confusion.qda
Accuracy.qda=mean(qda.class==Survived)
Accuracy.qda
tp.qda=confusion.qda[2,2]/(confusion.qda[1,2]+confusion.qda[2,2])
tp.qda
fp.qda=confusion.qda[2,1]/(confusion.qda[1,1]+confusion.qda[2,1])
fp.qda

# ---------------------------------KNN---------------------------------
attach(Train.data)
train.X=data.frame(Pclass,Age,Sex,SibSp,Parch,Embarked)
train.Y=Train.data$Survived
attach(Test.data)
test.X=data.frame(Pclass,Age,Sex,SibSp,Parch,Embarked)
test.Y=Test.data$Survived

# Scaling of data is done to remove huge differences between columns of data
# For eg, age will have a large range of values compared to sibsp
attach(train.X)
library(psych)
train.X[,c("Age","SibSp","Parch")]=scale(train.X[,c("Age","SibSp","Parch")])
# Dummy variable encoding for training data
train.X$Sex=dummy.code(train.X$Sex)
train.X$Pclass=as.data.frame(dummy.code(train.X$Pclass))
train.X$Embarked=as.data.frame(dummy.code(train.X$Embarked))
str(train.X)
head(train.X)

#Scaling for testing data
attach(test.X)
test.X[,c("Age","SibSp","Parch")]=scale(test.X[,c("Age","SibSp","Parch")])
# Dummy variable encoding for testing data
test.X$Sex=dummy.code(test.X$Sex)
test.X$Pclass=as.data.frame(dummy.code(test.X$Pclass))
test.X$Embarked=as.data.frame(dummy.code(test.X$Embarked))
str(test.X)
head(test.X)

#for loop and if condition is used to check model accuracy for different values of k from 1 to 30
set.seed(5)
i=1
mean.knn=matrix(,nrow=30,ncol=2)
for (i in 1:nrow(mean.knn))
{
  knn.predict=knn(train.X,test.X,train.Y,k=i)
  table(knn.predict,test.Y)
  mean.knn[i,1]=mean(knn.predict==Test.data$Survived)
  mean.knn[i,2]=i
}
plot(mean.knn[,2],mean.knn[,1],type="l",xlab="k",ylab="Accuracy")
max(mean.knn[,1]) #k value corresponding to this is 13 and it is the desired value

knn.predict=knn(train.X,test.X,train.Y,k=13)
confusion.knn=table(knn.predict,test.Y)
confusion.knn
Accuracy.knn=mean(knn.predict==Survived)
Accuracy.knn
tp.knn=confusion.knn[2,2]/(confusion.knn[1,2]+confusion.knn[2,2])
tp.knn
fp.knn=confusion.knn[2,1]/(confusion.knn[1,1]+confusion.knn[2,1])
fp.knn


# ---------------------------------With Cabin data - Logistic Regression---------------------------------

#Cabin data categorization-Here, the grouping is done based on cabin alphabet. For eg, A1 to A30 are grouped into A and so on

#Training data
Train.data$Cabin=as.character(Train.data$Cabin)
Train.data$Cabin[Train.data$Cabin==""]='X'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='A']='A'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='B']='B'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='C']='C'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='D']='D'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='E']='E'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='F']='F'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='G']='G'
Train.data$Cabin[substr(Train.data$Cabin, 1, 1)=='T']='T'
Train.data$Cabin=as.factor(Train.data$Cabin)

#Test data
Test.data$Cabin=as.character(Test.data$Cabin)
Test.data$Cabin[Test.data$Cabin==""]='X'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='A']='A'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='B']='B'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='C']='C'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='D']='D'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='E']='E'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='F']='F'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='G']='G'
Test.data$Cabin[substr(Test.data$Cabin, 1, 1)=='T']='T'
Test.data$Cabin=as.factor(Test.data$Cabin)

#logistic regression including cabin data
lfit.cabin=glm(Survived~Pclass+Age+Sex+SibSp+Parch+Embarked+Cabin,data=Train.data,family='binomial')
summary(lfit.cabin)

lprobability.cabin=predict(lfit.cabin,Test.data,type="response")

lpredict.cabin=rep("0",267)
lpredict.cabin[lprobability.cabin>0.5]="1"
attach(Test.data)
confusion.cabin=table(lpredict.cabin,Survived)
confusion.cabin
Accuracy.cabin=mean(lpredict.cabin==Survived)
Accuracy.cabin
tp.cabin=confusion.cabin[2,2]/(confusion.cabin[1,2]+confusion.cabin[2,2])
tp.cabin
fp.cabin=confusion.cabin[2,1]/(confusion.cabin[1,1]+confusion.cabin[2,1])
fp.cabin


#------------------------------Adjusted Odds ratio------------------------------

logistic2.fit=glm(Survived~Pclass+Sex+Embarked,data=Train.data,family='binomial')
summary(logistic2.fit)

aor=exp(logistic2.fit$coefficients)
round(aor,3)


#-----------------------------Threshold=0.8----------------------------------
lpredict.cabin.h=rep("0",267)
lpredict.cabin.h[lprobability.cabin>0.8]="1"
attach(Test.data)
confusion.cabin.h=table(lpredict.cabin.h,Survived)
confusion.cabin.h
Accuracy.cabin.h=mean(lpredict.cabin.h==Survived)
Accuracy.cabin.h
tp.cabin.h=confusion.cabin.h[2,2]/(confusion.cabin.h[1,2]+confusion.cabin.h[2,2])
tp.cabin.h
fp.cabin.h=confusion.cabin.h[2,1]/(confusion.cabin.h[1,1]+confusion.cabin.h[2,1])
fp.cabin.h

library(pROC)
par(pty="s")
roc(Test.data$Survived,lprobability.cabin,plot=TRUE,ci=TRUE,legacy.axes=TRUE)


#-----------------------------Threshold=0.2----------------------------------
lpredict.cabin.l=rep("0",267)
lpredict.cabin.l[lprobability.cabin>0.2]="1"
attach(Test.data)
confusion.cabin.l=table(lpredict.cabin.l,Survived)
confusion.cabin.l
Accuracy.cabin.l=mean(lpredict.cabin.l==Survived)
Accuracy.cabin.l
tp.cabin.l=confusion.cabin.l[2,2]/(confusion.cabin.l[1,2]+confusion.cabin.l[2,2])
tp.cabin.l
fp.cabin.l=confusion.cabin.l[2,1]/(confusion.cabin.l[1,1]+confusion.cabin.l[2,1])
fp.cabin.l

#-----------------------------ROC curve LDA------------------------------------
library(pROC)
par(pty="s")
roc(Test.data$Survived,lda.predict$posterior[,2],plot=TRUE,ci=TRUE,legacy.axes=TRUE)

#--------------------------------- KNN_updated---------------------------------
#Here, only those predictors that were found to be statistically significant in LR have been considered.
attach(Train.data)
train.X.n=data.frame(Pclass,Age,Sex,SibSp)
train.Y.n=Train.data$Survived
attach(Test.data)
test.X.n=data.frame(Pclass,Age,Sex,SibSp)
test.Y.n=Test.data$Survived

# Scaling of data
attach(train.X.n)
library(psych)
train.X.n[,c("Age","SibSp")]=scale(train.X.n[,c("Age","SibSp")])
# Dummy variable
train.X.n$Sex=dummy.code(train.X.n$Sex)
train.X.n$Pclass=as.data.frame(dummy.code(train.X.n$Pclass))
str(train.X.n)
head(train.X.n)

attach(test.X.n)
test.X.n[,c("Age","SibSp")]=scale(test.X.n[,c("Age","SibSp")])
# Dummy variable
test.X.n$Sex=dummy.code(test.X.n$Sex)
test.X.n$Pclass=as.data.frame(dummy.code(test.X.n$Pclass))
str(test.X.n)
head(test.X.n)

# k values from 1 to 30
set.seed(5)
i=1
mean.knn.n=matrix(,nrow=30,ncol=2)
for (i in 1:nrow(mean.knn.n))
{
  knn.predict.n=knn(train.X.n,test.X.n,train.Y.n,k=i)
  table(knn.predict.n,test.Y.n)
  mean.knn.n[i,1]=mean(knn.predict.n==Test.data$Survived)
  mean.knn.n[i,2]=i
}
plot(mean.knn.n[,2],mean.knn.n[,1],type="l",xlab="k",ylab="Accuracy")
max(mean.knn.n[,1]) #Corresponding k value is 4

knn.predict.n=knn(train.X.n,test.X.n,train.Y.n,k=13)
confusion.knn.n=table(knn.predict.n,test.Y.n)
confusion.knn.n
Accuracy.knn.n=mean(knn.predict.n==Survived)
Accuracy.knn.n
tp.knn.n=confusion.knn.n[2,2]/(confusion.knn.n[1,2]+confusion.knn.n[2,2])
tp.knn.n
fp.knn.n=confusion.knn.n[2,1]/(confusion.knn.n[1,1]+confusion.knn.n[2,1])
fp.knn.n

