setwd("E:/Google Drive/Data Science/Coursera Data Science/Course 8")

#Load Libraries
library(dplyr)
library(tidyr)
library(ggplot2)
library(caret)
library(tibble)
library(plotly)
library(reshape2)
library(corrplot)
library(parallel)
library(doParallel)

#Aquire Data
setwd("E:/Google Drive/Data Science/Coursera Data Science/Course 8")
fileURL.main <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
fileURL.pred <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(fileURL.main, destfile = "mainData.csv")
download.file(fileURL.pred, destfile = "predData.csv")

mainData <- read.csv("mainData.csv") %>% as_tibble()
predData <- read.csv("predData.csv") %>% as_tibble()

#review file structures
namesComp <- names(mainData) == names(predData) 
summary(namesComp)
which(namesComp == FALSE)

nameDiff <- c(head(mainData, 5)[,160],head(predData, 5)[,160])
nameDiff %>% as.data.frame()


##review data types
str(mainData)
dataType.main <- sapply(mainData, class) %>% 
        as.data.frame() %>% 
        rownames_to_column()

str(predData)
dataType.pred <- sapply(predData, class) %>% 
        as.data.frame() %>% 
        rownames_to_column()


#amend data types
mainData.num <- mainData
mainData.num[, 8:159] <- sapply(mainData.num, function(x) as.numeric(as.character(x)))

predData.num <- predData
predData.num[, 8:159] <- sapply(predData.num, function(x) as.numeric(as.character(x)))


#check for NAs
ggplot_missing <- function(x){
        x %>% is.na %>% melt %>% 
                ggplot(data = ., aes(x = Var2, y = Var1)) +
                geom_raster(aes(fill = value)) +
                scale_fill_manual(name = "", values = c("#D3D3D3", "#910000"), labels = c("Info", "NA")) +
                scale_y_reverse() +
                xlab("Column") + 
                ylab("Row") + 
                ggtitle("Missing Values Map") +
                theme_minimal() +
                theme(axis.text.x=element_blank(),
                      axis.ticks.x=element_blank(),
                      legend.position="bottom")
}
missingMap.main <- ggplot_missing(mainData.num)
missingMap.pred <- ggplot_missing(predData.num)

missingMap.main
missingMap.pred

#removing NAs from training set
mainDataClean <- mainData.num
mainDataClean <- mainDataClean %>% select_if(~ !any(is.na(.)))

predDataClean <- predData.num
predDataClean <- predDataClean %>% select_if(~ !any(is.na(.)))


#collate remaining shared features
mainCols <- names(mainDataClean)
predCols <- names(predDataClean)

sharedCols <- intersect(mainCols, predCols)
mainDataShared <- mainDataClean[,sharedCols]
mainDataShared <- cbind(mainDataShared, mainDataClean[,57])

#check for outliers
numericVars <- Filter(is.double, mainDataShared) 
outliers <- numericVars %>% sapply(function(x) boxplot(x, plot=FALSE)$out)
outliersAmount <- summary(outliers) %>% as.data.frame()
outliersAmount <- outliersAmount %>% filter(Var2 == "Length") 
outliersAmount$n <- as.numeric(as.character(outliersAmount$Freq))
outliersAmount <- outliersAmount%>% mutate(percent = n/nrow(numericVars)*100)

g1 <- ggplot(outliersAmount, aes(Var1, percent)) +
        geom_col(fill = "#779edd") +
        theme_bw() +
        xlab("Feature") + 
        ylab("% Outliers") + 
        ggtitle("% Outlier Per Numeric Feature") +
        theme(axis.text.x=element_blank(),
              axis.ticks.x=element_blank())
g1 <- ggplotly(g1)
g1

#near zero variance
nzv <- mainDataShared %>% nearZeroVar(saveMetrics = TRUE) %>% rownames_to_column()
nzvRemove <- nzv %>% filter(nzv == TRUE)
nzvRemove <- nzvRemove[1,1] 
mainDataShared <- mainDataShared %>% select(-nzvRemove)

#correlation & colinearity
corrMatrix <- numericVars %>% cor()
highCorr <- findCorrelation(corrMatrix, cutoff = 0.85)
exploreData <- mainDataShared[,-highCorr]

#remove timestamp/ID features features
finalData <- exploreData %>% select(-X, -user_name, -raw_timestamp_part_1, -raw_timestamp_part_2, -roll_belt)

###MODELLING


#create training/testing sets
set.seed(111)
trainIndex <- createDataPartition(finalData$classe, p = 0.75, list = FALSE)
training <- finalData[ trainIndex,]
testing  <- finalData[-trainIndex,]

#Set up parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)

#Cross Validation 5 fold
fitControl<- trainControl(method = "cv", number = 5, savePredictions = TRUE, allowParallel = TRUE)


###SVM
#modelling SVM
system.time(model.SVM.classe <- train(classe ~ ., data = training, method = "svmLinear", trControl = fitControl))
model.SVM.classe

#Predictions on the test set (model = model name, testing = test set)
predictions.SVM.classe <- predict(model.SVM.classe, testing)
testing$predictions.SVM.classe <- predictions.SVM.classe 
postResample(testing$predictions.SVM.classe, testing$classe)

#Confusion matrix
confMatrix.SVM <- confusionMatrix(testing$classe, testing$predictions.SVM.classe)
confMatrix.SVM$table

###Random Forest
#modelling RF
system.time(model.RF.classe <- train(classe ~ ., data = training, method = "rf", trControl = fitControl))
model.RF.classe

#Predictions on the test set (model = model name, testing = test set)
predictions.RF.classe <- predict(model.RF.classe, testing)
testing$predictions.RF.classe <- predictions.RF.classe 

#Confusion matrix
confMatrix.RF <- confusionMatrix(testing$classe, testing$predictions.RF.classe)
confMatrix.RF$table
postResample(testing$predictions.RF.classe, testing$classe)

###DECISION TREE
#modelling DT
system.time(model.DT.classe <- train(classe ~ ., data = training, method = "rpart", trControl = fitControl))
model.DT.classe

#Predictions on the test set (model = model name, testing = test set)
predictions.DT.classe <- predict(model.DT.classe, testing)
testing$predictions.DT.classe <- predictions.DT.classe 

#Confusion matrix
confMatrix.DT <- confusionMatrix(testing$classe, testing$predictions.DT.classe)
confMatrix.DT$table
postResample(testing$predictions.DT.classe, testing$classe)


###Gradient Boosting Machine
#modelling GBM
system.time(model.GBM.classe <- train(classe ~ ., data = training, method = "gbm", trControl = fitControl, verbose = FALSE))
model.GBM.classe

#Predictions on the test set (model = model name, testing = test set)
predictions.GBM.classe <- predict(model.GBM.classe, testing)
testing$predictions.GBM.classe <- predictions.GBM.classe 

#Confusion matrix
confMatrix.GBM <- confusionMatrix(testing$classe, testing$predictions.GBM.classe)
confMatrix.GBM$table
postResample(testing$predictions.GBM.classe, testing$classe)

##Turn off parallel processing
stopCluster(cluster)
registerDoSEQ()

#prediction exercise
predictions.exercise <- predict(model.RF.classe, predDataClean)
predDataClean$classe <- predictions.exercise
predictions.exercise