#############################################
# MAIN SCRIPT TO RUN FOR EVIDENTIAL BAGGING #
#############################################

# Variables and screen cleaning:
graphics.off()   # clear plots
cat("\014")      # clear the console
rm(list = ls())  # clear memory
options(warn=-1) # turns off warnings

# Directory path:
windowsPath <- "D:/A_Sauver/Recherche/Articles/Mes articles/Conferences/IPMU 2018/Experiments/"
ubuntuPath  <- "/media/bob/B60E0F450E0EFDDD/Users/toto/Documents/Recherche/Articles/Mes articles/Conferences/IPMU 2018/Experiments/"
setwd(ubuntuPath)
source("myLib.R")
require(rpart)
require(randomForest)
require(e1071)
require(nnet)
require(MASS)
require(ggplot2)
require(ggthemes)

# INPUTS:
#### Dataset names:
# "contraceptiveMethod", "glass", "balanceScale", "wine", "banknote", "occupancy", "banana", "mammographic", 
# "pima", "nursery", "satimage", "ticTacToe", "titanic", "iris", "ecoli2", "breastTissue"
#
# * more difficult or less interesting: breastTissue, mushrooms, creditCardDefault, thyroid, letter, magic, shuttle
datasetNames       <- c("contraceptiveMethod", "glass", "balanceScale", "wine", "banknote", "occupancy", "banana", "mammographic", 
                        "pima", "nursery", "satimage", "ticTacToe", "titanic", "iris", "ecoli2", "breastTissue")
#### Learning algorithms (EBag1 = Pmean, EBag2 = Ebag_SD, EBag3 = Ebag_CD, EBag4 = Ebag_con):
# "tree", "forest", SVM", "nnet", "bayes", "lda", "voteBag", "EBag1", "EBag2", "EBag3", "EBag4"
learningAlgorithms <- c("tree", "forest", "SVM", "nnet", "bayes", "lda", "voteBag", "EBag1", "EBag2", "EBag3", "EBag4")
nRuns              <- 2
nModelPerBag       <- 1
nnetSize           <- 10
combinationType    <- "Mean" # from c("Mean", "Dempster", "Disjunction") 

# Runs:
str <- paste0("./Results/plotResults_", Sys.time(), ".jpg")
str <- gsub(" ", "_", str)
str <- gsub(":", "-", str)
jpeg(str, width = 2000, height = 1000)
if (length(datasetNames) > 1){
  par(mfrow = c(2, ceiling(length(datasetNames)/2)), oma = c(5,5,5,5))
}

finalResults <- c()
for (iDataset in 1 : length(datasetNames)){
  
    datasetName <- datasetNames[iDataset]
    cat("dataset", datasetName, "\n\n")
    
    # Data mport:
    setwd("./Datasets/")
    eval(parse(text = paste0("dataset <- as.data.frame(read.csv2('", datasetName, ".csv'))")))
    setwd("..")
    
    # Data preprocessings:
    J <- ncol(dataset) - 1
    for (varName in setdiff(names(dataset), 'class')){
      dataset[, varName] <- as.numeric(dataset[, varName]) # for each attribute conversion into numerical format
    }
    
    # RUN:
    results1dataset      <- evidBagging(dataset, learningAlgorithms, nClassifierPerBag, nRuns, nnetSize, nModelPerBag, combinationType)
    results1datasetSmall <- subset(results1dataset, select = grepl("Bag", names(results1dataset))) # keep only bagging results (not single classifiers' ones)
    finalResults         <- rbind(finalResults, cbind(dataset  = rep(datasetName, nRuns),     #
                                                      combType = rep(combinationType, nRuns), # results 
                                                      bagSize  = rep(nModelPerBag, nRuns),    # formating
                                                      results1dataset))                       #
    
    # Results printing:
    if (length(datasetNames) > 1){
      boxplot(results1datasetSmall, main = datasetName, cex.main = 3, cex.axis = 2)
    } else {
      boxplot(results1datasetSmall, main = datasetName, cex.main = 3, cex.axis = 2, oma = c(5,5,5,5))
    }
}

str <- paste(combinationType, "'s combination of ", nModelPerBag, "-bags, ", nRuns, " runs")
title(str, outer=TRUE, adj = 0, col.main = "red", line = 0, cex.main = 3)
dev.off()

# Results saving:
setwd("./Results/")
str <- paste0("results_", Sys.time(), ".csv")
str <- gsub(" ", "_", str)
str <- gsub(":", "-", str)
write.csv2(finalResults, str, sep = ",")
MEANresults <- NULL
for (iDataset in 1 : length(datasetNames)){
  finalResults1dataset <- finalResults[finalResults$dataset == datasetNames[iDataset], ]
  MEANresults          <- rbind(MEANresults, colMeans(finalResults1dataset[, 4 : ncol(finalResults1dataset)]))
}
MEANresults <- cbind(datasetNames, as.data.frame(MEANresults))
str <- paste0("MEANresults_", Sys.time(), ".csv")
str <- gsub(" ", "_", str)
str <- gsub(":", "-", str)
write.csv2(MEANresults, str)
setwd("..")
print.data.frame(MEANresults, digits = 3)
print(round(colMeans(MEANresults[, 2 : ncol(MEANresults)]), 3))

# 2 classes datasets results:
cat("\n2 classes datasets results:\n")
twoClassesDatasets <- c("banknote", "occupancy", "banana", "mammographic", "pima", "ticTacToe", "titanic")
twoClassesMEANresults <- MEANresults[MEANresults$datasetNames %in% twoClassesDatasets, ]
print(round(colMeans(twoClassesMEANresults[, 2 : ncol(twoClassesMEANresults)]), 3))

# more than 2 classes datasets results:
cat("\nmore than 2 classes datasets results:\n")
moreClassesDatasets <- setdiff(datasetNames, c("banknote", "occupancy", "banana", "mammographic", "pima", "ticTacToe", "titanic"))
moreClassesMEANresults <- MEANresults[MEANresults$datasetNames %in% moreClassesDatasets, ]
print(round(colMeans(moreClassesMEANresults[, 2 : ncol(moreClassesMEANresults)]), 3))

