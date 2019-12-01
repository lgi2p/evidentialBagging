#############################################
# MAIN SCRIPT TO RUN FOR EVIDENTIAL BAGGING #
#############################################

library(rpart)
library(e1071)
library(randomForest)
library(nnet)
library(MASS)

# Variables and screen cleaning:
graphics.off()   # clear plots
cat("\014")      # clear the console
rm(list = ls())  # clear memory
options(warn=-1) # turns off warnings

# Directory path:
windowsPath <- "D:/Google Drive/Recherche/Publications/Conferences/IPMU 2018/Experiments/"
setwd(windowsPath)
source("myLib.R")

# INPUTS:
#### datasetNames from ("contraceptiveMethod", "glass", "balanceScale", "wine", "banknote", 
#                       "occupancy", "banana", "mammographic", "pima", "nursery", "satimage", 
#                       "ticTacToe", "titanic", "iris", "ecoli2", "breastTissue")
#            * more difficult or less interesting: breastTissue, mushrooms, creditCardDefault, thyroid, letter, magic, shuttle
# fast dataset: balanceScale, banknote (but pb), mammographic (but pb), banana, pima
# slow datasets: contraceptiveMethod, occupancy, nursery, satimage
# new problems on: glass, wine, banknote, banana
datasetNames       <- c("nursery", "satimage", "occupancy", "contraceptiveMethod", "balanceScale", "pima")
#### learningAlgorithms from c("tree", "forest", SVM", "nnet", "bayes", "lda", 
#                              "voteBag", "EBag1", "EBag2", "EBag3", "EBag4"
learningAlgorithms <- c("tree", "forest", "SVM", "nnet", "bayes", "lda", "voteBag", "EBag1", "EBag2", "EBag3", "EBag4") # EBag1 = mean, EBag2 = simple discounting, EBag3 = class-dependant, EBag4 = contextual
nRuns              <- 20
nFold              <- 10
nModelPerBag       <- 1
nnetSize           <- 10
combinationType    <- "Mean" # from c("Mean", "Dempster", "Disjunction") 
parms <- list(learningAlgorithms = learningAlgorithms, nRuns = nRuns, nFold = nFold, 
              nModelPerBag = nModelPerBag, nnetSize = nnetSize, combinationType = combinationType)

# Runs:
# setwd("./Results/")
str <- paste0("./Results/plotResults_", Sys.time(), ".jpg")
str <- gsub(" ", "_", str)
str <- gsub(":", "-", str)
jpeg(str, width = 2000, height = 1000)
if (length(datasetNames) > 1){
  par(mfrow = c(2, ceiling(length(datasetNames)/2)), oma = c(5,5,5,5))
}
# setwd("..")

iDataset <- 1

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
    results1dataset      <- evidBagging(dataset, parms)
    results1datasetSmall <- subset(results1dataset, select = grepl("Bag", names(results1dataset)))
    finalResults         <- rbind(finalResults, cbind(dataset  = rep(datasetName, nRuns), 
                                                      combType = rep(combinationType, nRuns), 
                                                      bagSize  = rep(nModelPerBag, nRuns), 
                                                      results1dataset))
    
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
write.table(finalResults, str, row.names = F, dec = ".", sep = ";")
MEANresults <- NULL
for (iDataset in 1 : length(datasetNames)){
  finalResults1dataset <- finalResults[finalResults$dataset == datasetNames[iDataset], ]
  MEANresults          <- rbind(MEANresults, colMeans(finalResults1dataset[, 4 : ncol(finalResults1dataset)]))
}
MEANresults <- cbind(datasetNames, as.data.frame(MEANresults))
str <- paste0("MEANresults_", Sys.time(), ".csv")
str <- gsub(" ", "_", str)
str <- gsub(":", "-", str)
write.table(MEANresults, str, row.names = F, dec = ".", sep = ";")
setwd("..")
print.data.frame(MEANresults, digits = 3)


