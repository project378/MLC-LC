

#include libraries 

library(mlr)
library(mldr)            
library(utiml)           # contains all the algorithems (RAkEL, LP, BR)
library(mldr.datasets)   # All the datasets
library(rpart)		       # Decision tree 
library(e1071)		       # For SVM
library(C50)		 

enron()			             # Import dataset

set <- NULL

set0 <- colnames(enron$dataset)[enron$labels$index] # all the label set

# Cluster partition sets (Output of C Cube clustering)
set[[1]] <- c("A.A1","C.C9","C.C11","C.C7","B.B3","D.D19","B.B8","D.D12","C.C8","B.B9","C.C10","B.B11","D.D2","D.D15","C.C4","B.B6","D.D13","C.C12","B.B5","D.D11","C.C3","D.D18","B.B10","D.D14")
set[[2]] <- c("B.B2","A.A8","B.B12","D.D16","D.D1","D.D6","D.D3","A.A2","D.D4")
set[[3]] <- c("A.A3","D.D5","D.D8","D.D7","D.D17")
set[[4]] <- c("A.A4","C.C13")
set[[5]] <- c("B.B1","D.D9")
set[[6]] <- c("B.B13","A.A7")
set[[7]] <- c("A.A6","B.B7")
set[[8]] <- c("C.C6","C.C1","D.D10","C.C2","B.B4","A.A5","C.C5")   # all the single labeled clusters for BR



y <- NULL
yy <- NULL
yyf <- NULL
yyl <- NULL


lblIndex <- 1002:1055		# label indexes in the dataset
brIndex <- 8

# Form a new column with column heading with column name "Class" and set to 0.
x1 <- enron$dataset[1002]
colnames(x1)[1]<- c("Class")
x1[,1] <- 0


for(i in 1:length(set)){
  yy[[i]] <- enron$dataset
  yy[[i]][,setdiff(set0,set[[i]])] <- 0			# set all the other columns not in the cluster partition to 0.
  
  yyf[[i]] <- yy[[i]][,enron$attributesIndexes]
  yyl[[i]] <- yy[[i]][,enron$labels$index]
  
  for(k in 1:nrow(yyl[[i]])){				        # if a row contains all labels absent, then new column set to 1.
    if(rowSums(yyl[[i]][k,]) < 1){
      x1[k,1] <- 1
    }
  }
  
  yy[[i]] <- cbind(yyf[[i]],yyl[[i]],x1)		# Form new dataset with feature vector, label vector and new column
  x1[,1] <- 0
  
  y[[i]] <- mldr_from_dataframe(yy[[i]], labelIndices = lblIndex)	# create a 'mldr' object
}

# results write to a file
sink("enron.txt")

# same process runs 10 iterations with random holdout cross validation
for(gg in 1:10){
  
  y0 <- create_holdout_partition(y[[1]], c(train=0.66, test=0.34), "random")	# separate dataset in to Test and Training randomly
  train <- row.names(y0$train$dataset)
  test <- row.names(y0$test$dataset)
  
  y1Train <- NULL
  y1Test <- NULL
  lpMod <- NULL
  lpPred <- NULL
  conf <- NULL
  
  for(i in 1:length(set)){
    y1Train[[i]] <- mldr_from_dataframe(y[[i]]$dataset[train,], labelIndices = lblIndex)
    y1Test[[i]] <- mldr_from_dataframe(y[[i]]$dataset[test,], labelIndices = lblIndex)
    
    if(i == brIndex){
      lpMod[[i]] <- br(y1Train[[i]],"C5.0")			# Binary relevance for single labeled clusters
    } else{
      lpMod[[i]] <- lp(y1Train[[i]],"C5.0")			# Label power set (LP) for all the other clusters
    }
    lpPred[[i]] <- predict(lpMod[[i]],y1Test[[i]])		# Prediction step
    conf[[i]] <- multilabel_confusion_matrix(y1Test[[i]],lpPred[[i]])	# Confusion matrix for the prediction
  }
  
  
  TP <- 0
  FP <- 0
  TN <- 0
  FN <- 0
  
  # Evauation
  for(k in 1:length(set)){
    pClus1 <- as.bipartition(lpPred[[k]])
    pClus1 <- pClus1[,set[[k]]]			# predicted values only for current partition
    tClus1 <- y1Test[[k]]$dataset[,set[[k]]]	# Test values only for current partition
    
    for(j in 1:ncol(pClus1)){			# collect True Positive(TP), True Negative(TN), False Positive(FP) and False Negative (FN)
      for(i in 1:nrow(pClus1)){
        if(tClus1[i,j] == 1 && pClus1[i,j] == 1){
          TP <- TP + 1
        }
        if(tClus1[i,j] == 1 && pClus1[i,j] == 0){
          FN <- FN + 1
        }
        if(tClus1[i,j] == 0 && pClus1[i,j] == 1){
          FP <- FP + 1
        }
        if(tClus1[i,j] == 0 && pClus1[i,j] == 0){
          TN <- TN + 1
        }
      }
      cat("|",TP," ",FP," ",FN," ",TN,"|\n")	# write all to a file, to calculate Micro and Macro F1 measure
      TP <- 0
      FP <- 0
      TN <- 0
      FN <- 0
    }
  }
  cat("----\n")
}

# File close
sink()

# I have used simple java programe to calculate F1 measures from this out file.

