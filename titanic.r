train <- read.csv("D://titanic//train.csv")

test <- read.csv("D://titanic/test.csv")

library(VIM)

aggr(train)
# age
train$Age <- impute(data.frame(train$Age),"median")


aggr(test)
#age and fare
test$Age <- impute(data.frame(test$Age), "median")
test$Fare <- impute(data.frame(test$Fare),"median")

train.survived <- train$Survived

all <- rbind(train[,-2],test)

all$Parch <- as.factor(all$Parch)
train<- all[1:891,]
train<-cbind(data.frame(train.survived),train)

normalise <- function(x) {
  return((x-min(x))/(max(x)-min(x)))
}

PassengerID <- test$PassengerId

train$Age <- normalise(train$Age)
train$Fare <- normalise(train$Fare)
train$PassengerId <- normalise(train$PassengerId)

test$Age <- normalise(test$Age)
test$Fare <- normalise(test$Fare)

test$Parch <- as.factor(test$Parch)
test$PassengerId <- normalise(test$PassengerId)

library(nnet)

# 
# nnet(x, y, weights, size, Wts, mask,
#      linout = FALSE, entropy = FALSE, softmax = FALSE,
#      censored = FALSE, skip = FALSE, rang = 0.7, decay = 0,
#      maxit = 100, Hess = FALSE, trace = TRUE, MaxNWts = 1000,
#      abstol = 1.0e-4, reltol = 1.0e-8, ...)


set.seed(77)
train.nn <- nnet(as.factor(train.survived)~as.factor(Pclass)+Sex+as.factor(Parch)+Embarked+Age+as.factor(SibSp)+Fare+PassengerId, data = train,size = 12,MaxNWts = 1727, maxit=2000, rang= 1)

train.nn <- nnet(as.factor(train.survived)~as.factor(Pclass)+Embarked+Sex+Parch+
                   Age+as.factor(SibSp)+as.factor(Parch)+as.matrix(Fare), data = train,size = 5,MaxNWts = 1727, 
                 maxit=2000000, rang= 1,decay = 5e-6,  abstol = 1.0e-4, reltol = 1.0e-8)

weights <- train.nn$wts # 190.56

pred <- predict(train.nn,test,type = "class")
fit <- predict(train.nn,train,type = "class")

nn.table<-table(train$Survived,fit)

library(caret)
confusionMatrix(nn.table)


Survived <- pred

output <- cbind.data.frame(PassengerID,Survived)

write.csv(output,"D://titanic.csv",row.names = FALSE)




