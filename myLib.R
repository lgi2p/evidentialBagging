baggingResampling <- function(learningDataset, allAtt = T){
  
  classLabels     <- unique(learningDataset$class)
  classLabels     <- classLabels[order(classLabels)]
  N               <- nrow(learningDataset)
  J               <- ncol(learningDataset) - 1
  K               <- length(classLabels)
  learningDataset <- learningDataset[sample(1 : N), ]
  
  if (allAtt == T){
    baggingDataset <- c()
    for (iLabel in 1 : K){
      firstLabelledIndexes <- which(learningDataset$class == classLabels[iLabel])
      baggingDataset       <- rbind(baggingDataset, learningDataset[firstLabelledIndexes[1], ])
      learningDataset      <- learningDataset[- firstLabelledIndexes[1], ]
    }
    N2             <- nrow(learningDataset)
    baggingDataset <- learningDataset[sample(1 : N2, size = N2, replace = TRUE), ]
  } else {
    nSelectedAtt       <- sample(3 : J, size = 1)
    selectedAttIndexes <- sample(1 : J, size = nSelectedAtt)
    selectedAttIndexes <- selectedAttIndexes[order(selectedAttIndexes)]
    baggingDataset <- c()
    for (iLabel in 1 : K){
      firstLabelledIndexes <- which(learningDataset$class == classLabels[iLabel])
      baggingDataset       <- rbind(baggingDataset, learningDataset[firstLabelledIndexes[1], selectedAttIndexes])
      learningDataset      <- learningDataset[- firstLabelledIndexes[1], ]
    }
    N2             <- nrow(learningDataset)
    baggingDataset <- learningDataset[sample(1 : N2, size = N2, replace = TRUE), c(selectedAttIndexes, J + 1)]
  }
  
  baggingDataset$class <- factor(baggingDataset$class)
  
  return(baggingDataset)
}
#############################################################################################
class2probaPred <- function(classPredictions, levels){
  probaPredictions <- c()
  for (i in 1 : nrow(classPredictions)){
    prob             <- classPredictions[i, ]/sum(classPredictions[i, ])
    probaPredictions <- rbind(probaPredictions, prob)
  }
  rownames(probaPredictions) <- NULL
  colnames(probaPredictions) <- levels

  return(probaPredictions)
}
#############################################################################################
evidBagging <- function(dataset, parms){
  
  learningAlgorithms = parms$learningAlgorithms; nClassifierPerBag = parms$nClassifierPerBag; nRuns = parms$nRuns
  nFold = parms$nFold; nnetSize = parms$nnetSize; nModelPerBag = parms$nModelPerBag; combinationType = parms$combinationType
  equation    <- "class ~ ."
  classLabels <- unique(dataset$class)
  classLabels <- classLabels[order(classLabels)]
  singleClassifierNames <- setdiff(learningAlgorithms, 
                                   learningAlgorithms[!grepl("Bag", learningAlgorithms)])
  
  for (iClassifier in 1 : length(learningAlgorithms)){
    eval(parse(text = paste0(learningAlgorithms[iClassifier], "FinalAcc <- c()")))
  }
  iRun <- 1
  iFold <- 1
  
  compTimes <- c()
  for (iRun in 1 : nRuns){
    cat("RUN", iRun, "/", nRuns, "\n\n")
    CVfoldsIndexes <- createFoldsME(dataset$class)
    for (iClassifier in 1 : length(learningAlgorithms)){
      eval(parse(text = paste0(learningAlgorithms[iClassifier], "Accuracies <- c()")))
    }

    CT <- 0
    for (iFold in 1 : nFold){
      if (iFold < 10){
        cat("    fold ", iFold, "/", nFold, ": ")
      } else {
        cat("    fold", iFold, "/", nFold, ": ")
      }
      
      # Data sampling into learning, validation and testing sets:
      learningExamplesIndexes <- c()                                                           
      for (iLearningFold in eval(setdiff(1:nFold, iFold))){                                       
        learningExamplesIndexes <- c(learningExamplesIndexes, CVfoldsIndexes[[iLearningFold]]) 
      }                                                                                        
      learningData   <- dataset[learningExamplesIndexes,]                                      
      validationData <- learningData[eval(floor(0.75 * nrow(learningData)) + 1)  : nrow(learningData), ]
      learningData   <- learningData[1 : eval(floor(0.75 * nrow(learningData))), ]
      testingData    <- dataset[CVfoldsIndexes[[iFold]], ]
      trueClasses    <- testingData$class
      
      # Learning of single classifiers:
      compTimeLear   <- system.time(
        bags <- learnClassifier(equation, learningData, "Bag", learningAlgorithms, nModelPerBag)
      )
      compTimePred <- system.time(
        for (iClassifier in 1 : length(learningAlgorithms)){
          classifierName <- learningAlgorithms[iClassifier]
          if (grepl("Bag", classifierName)){
            cat(paste0(classifierName, ", "))
            
            # Predictions:
            classPredictions <- predictClass(classifierName, bags, validationData, testingData, learningAlgorithms, classLabels, nModelPerBag, combinationType)
            
            # Evaluation:
            classifierAcc <- sum(as.character(classPredictions) == as.character(trueClasses))/length(trueClasses)
            eval(parse(text = paste0(classifierName, "Accuracies <- c(", classifierName, "Accuracies, classifierAcc)")))
          }
        }
      )
      CT <- CT + compTimeLear[3] + compTimePred[3]
      cat("\n")
    }
    compTimes <- c(compTimes, CT)
    cat("\n")
    for (iClassifier in 1 : length(learningAlgorithms)){
      classifierName <- learningAlgorithms[iClassifier]
      if (grepl("Bag", classifierName)){
        eval(parse(text = paste0(classifierName, "FinalAcc <- c(", classifierName,
                                 "FinalAcc, mean(", classifierName, "Accuracies))")))
      }
    }
  }
  
  # Results formating:
  finalResults <- c()
  for (iClassifier in 1 : length(learningAlgorithms)){
    classifierName <- learningAlgorithms[iClassifier]
    if (grepl("Bag", classifierName)){
      classifierName <- learningAlgorithms[iClassifier]
      eval(parse(text = paste0("finalResults <- cbind(finalResults, ", classifierName, " = ",
                               classifierName, "FinalAcc)")))
    }
  }
  finalResults <- as.data.frame(finalResults)
  finalResults <- cbind(finalResults, CT = round(compTimes/60, 1))
  
  return(finalResults)
}
#############################################################################################
learnClassifier <- function(equation, learningData, classifierName, learningAlgorithms, nModelPerBag, bootstrap = T){
  
  
  learningData$class <- factor(as.character(learningData$class))
  if (prod(!grepl("Bag", learningAlgorithms))){
    bootstrap <- F
  }
  
  if (classifierName == "tree"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    classifier <- rpart(equation, learningData, method = "class")
  } else if (classifierName == "forest"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    str  <- "classifier <- randomForest(learningData[, setdiff(names(learningData), 'class') ], learningData$class, ntree = 20)"
    if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
      eval(parse(text = str))
    } else {
      if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
        eval(parse(text = str))
      } else {
        if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
          eval(parse(text = str))
        }
      }
    }
  } else if (classifierName == "bayes"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    classifier               <- naiveBayes(learningData[, setdiff(names(learningData), "class") ], learningData$class)
  } else if (classifierName == "SVM"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    classifier               <- eval(parse(text = paste0('svm(formula = ', equation, ', data = learningData, probability = T)')))
  } else if (classifierName == "nnet"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    classifier               <- eval(parse(text = paste0('nnet(formula = ', equation, ', data = learningData, size = nnetSize, trace = F)')))
  } else if (classifierName == "lda"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    classifier               <- lda(learningData[, setdiff(names(learningData), "class") ], learningData$class, tol = 1.0e-20)
  } else if (classifierName == "c.nnet"){
    if (bootstrap){
      learningData <- baggingResampling(learningData)
    }
    classifier               <- eval(parse(text = paste0("train(", equation, ", data = learningData, method='nnet', trControl=trainControl(method='cv'), trace = F)")))
  } else if (grepl("Bag", classifierName)){
    classifiers <- NULL
    for (iClassifier in 1 : length(learningAlgorithms)){
      if (!grepl("Bag", learningAlgorithms[iClassifier])){
        classifierName   <- learningAlgorithms[iClassifier]
        eval(parse(text = paste0("classifiers$", classifierName, " <- NULL")))
        for (iModel in 1 : nModelPerBag){
          BootStrappedlearningData <- baggingResampling(learningData)
          str  <- paste0("classifiers$", classifierName, "[[iModel]] <- learnClassifier(equation, BootStrappedlearningData, classifierName, learningAlgorithms)")
          if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
            eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
          } else {
            if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
              eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
            } else {
              if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
              } else {
                if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                  eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                } else {
                  if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                    eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                  } else {
                    if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                      eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                    } else {
                      if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                        eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                      } else {
                        if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                          eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                        } else {
                          if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                            eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                          } else {
                            if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                              eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                            } else {
                              if (!inherits(c <- try(eval(parse(text = str))),  "try-error")){
                                eval(parse(text = paste0("classifiers$", classifierName, "[[iModel]] <- c")))
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    classifier <- classifiers
  }
  
  return(classifier)
}
#############################################################################################
predictClass <- function(classifierName, classifier, validationData, testingData, learningAlgorithms, classLabels, nModelPerBag, combinationType){
  
  if (classifierName %in% c("tree", "forest", "bayes", "nnet")){
    classPredictions <- predict(classifier, testingData, type = "class")
  } else if (classifierName %in% c("c.nnet", "SVM")){
    classPredictions <- predict(classifier, testingData)
  } else if (classifierName %in% c("lda")){
    classPredictions <- predict(classifier, testingData[, setdiff(names(testingData), "class") ])$class
  } else if (classifierName == "voteBag"){
    classPredictions <- c()
    for (iClassifier in 1 : length(learningAlgorithms)){
      if (!grepl("Bag", learningAlgorithms[iClassifier])){
        singleClassifierName <- learningAlgorithms[iClassifier]
        for (iModel in 1 : nModelPerBag){
          eval(parse(text = paste0("singleClassifier <- classifier$", singleClassifierName, "[[iModel]]")))
          eval(parse(text = paste0("classPredictions <- cbind(classPredictions, ", singleClassifierName, 
                                   " = as.character(predictClass(singleClassifierName, singleClassifier, NULL, testingData, learningAlgorithms, classLabels)))")))
        }
      }
    }
    classPredictions <- vote(classPredictions)
  } else if (classifierName %in% c("EBag1", "EBag2", "EBag3", "EBag4")){
    evidPredPerClassifier <- bfGener(classifierName, classifier, validationData, testingData, classLabels, nModelPerBag)
    evidPred              <- bfFusion(classifierName, evidPredPerClassifier, nModelPerBag, combinationType)
    classPredictions      <- evidDecision(classifierName, evidPred)
  }

  return(classPredictions)
}
#############################################################################################
vote <- function(classPredictions){
  
  finalClassPredictions <- c()
  for (i in 1 : nrow(classPredictions)){
    finalClassPredictions <- c(finalClassPredictions, 
                               names(which.max(table(classPredictions[i, ]))))
  }
  
  return(finalClassPredictions)
}
#############################################################################################
bfGener <- function(classifierName, classifier, validationData, testingData, classLabels, nModelPerBag){
  
  if (classifierName == "EBag1"){
    evidClassPredictions <- NULL
    for (iClassifier in 1 : length(classifier)){
      singleClassifierName <- names(classifier)[iClassifier]
      oneModelBag          <- classifier[[iClassifier]]
      for (iModel in 1 : nModelPerBag){
        singleClassifier <- oneModelBag[[iModel]]
        
        # Probabilist predictions computation:
        classProbPred <- computeProbaPred(singleClassifierName, singleClassifier, testingData, classLabels)
        
        # Belief function generation:
        evidClassPredictions[[singleClassifierName]][[iModel]] <- classProbPred
      }
    }
  } else if (classifierName == "EBag2"){
    evidClassPredictions <- NULL
    for (iClassifier in 1 : length(classifier)){
      singleClassifierName <- names(classifier)[iClassifier]
      oneModelBag          <- classifier[[iClassifier]]
      for (iModel in 1 : nModelPerBag){
        singleClassifier <- oneModelBag[[iModel]]
        
        # Performance evaluation on test data:
        if (singleClassifierName == "c.nnet"){
          singleClassifierClassPred <- predict(singleClassifier, validationData)
        } else if (singleClassifierName == "lda"){
          singleClassifierClassPred <- predict(singleClassifier, validationData[, setdiff(names(validationData), "class")])$class
        } else {
          singleClassifierClassPred <- predict(singleClassifier, validationData, type = "class")
        }
        accuracy <- sum(as.character(singleClassifierClassPred) == as.character(validationData$class))/nrow(validationData)
        
        # Probabilist predictions computation:
        classProbPred <- computeProbaPred(singleClassifierName, singleClassifier, testingData, classLabels)
        
        # Belief function generation:
        classProbPred             <- classProbPred * accuracy
        classProbPred             <- cbind(classProbPred, Omega_Y = rep(1 - accuracy, nrow(testingData)))
        evidClassPredictions[[singleClassifierName]][[iModel]] <- classProbPred
      }
      
    }
  } else if (classifierName == "EBag3"){
    evidClassPredictions <- NULL
    for (iClassifier in 1 : length(classifier)){
      singleClassifierName <- names(classifier)[iClassifier]
      oneModelBag          <- classifier[[iClassifier]]
      for (iModel in 1 : nModelPerBag){
        singleClassifier <- oneModelBag[[iModel]]
        
        # Performance evaluation on test data:
        if (singleClassifierName == "c.nnet"){
          singleClassifierClassPred <- predict(singleClassifier, validationData)
        } else if (singleClassifierName == "lda"){
          singleClassifierClassPred <- predict(singleClassifier, validationData[, setdiff(names(validationData), "class")])$class
        } else {
          singleClassifierClassPred <- predict(singleClassifier, validationData, type = "class")
        }
        singleClassifierClassPred         <- factor(singleClassifierClassPred)
        levels(singleClassifierClassPred) <- classLabels
        levels(validationData$class)      <- classLabels
        confusionMatrix                   <- table(singleClassifierClassPred, validationData$class)
        
        # Probabilist predictions computation:
        classProbPred <- computeProbaPred(singleClassifierName, singleClassifier, testingData, classLabels)
        
        # Belief function generation:
        evidClassPredictions[[singleClassifierName]][[iModel]] <- proba2BF(classifierName, classProbPred, confusionMatrix)
      }
      
    }
  } else if (classifierName == "EBag4"){
    evidClassPredictions <- NULL
    for (iClassifier in 1 : length(classifier)){
      
      # Local miscalssification rate learning:
      singleClassifierName <- names(classifier)[iClassifier]
      oneModelBag          <- classifier[[iClassifier]]
      for (iModel in 1 : nModelPerBag){
        singleClassifier <- oneModelBag[[iModel]]
        
        # Performance evaluation on test data:
        if (singleClassifierName == "c.nnet"){
          singleClassifierClassPred <- predict(singleClassifier, validationData)
        } else if (singleClassifierName == "lda"){
          singleClassifierClassPred <- predict(singleClassifier, validationData[, setdiff(names(validationData), "class")])$class
        } else {
          singleClassifierClassPred <- predict(singleClassifier, validationData, type = "class")
        }
        df      <- subset(validationData, select = - class)
        df$diff <- as.numeric(as.character(singleClassifierClassPred) != as.character(validationData$class))
        if (max(df$diff) == 0 | length(unique(df$diff)) == 1){
          svm <- NULL
        } else {
          svm <- svm(diff ~ ., df)
        }
        
        # Probabilist predictions computation:
        classProbPred <- computeProbaPred(singleClassifierName, singleClassifier, testingData, classLabels)
        
        # Belief function generation:
        if (is.null(svm)){
          diff <- rep(0, nrow(testingData))
        } else {
          diff           <- predict(svm, testingData)
          diff[diff < 0] <- 0
        }
        
        classProbPred  <- classProbPred * (1 - diff)
        classProbPred  <- cbind(classProbPred, Omega_Y = diff)
        evidClassPredictions[[singleClassifierName]][[iModel]] <- classProbPred
      }
    }
  }
  
  return(evidClassPredictions)
}
#############################################################################################
bfFusion <- function(classifierName, evidPredPerClassifier, nModelPerBag, combinationType = "Mean"){
  if (combinationType == "Mean"){
    # finalBF <- evidPredPerClassifier[[1]]
    finalBF <- matrix(0, nrow = nrow(evidPredPerClassifier[[1]][[1]]), ncol = ncol(evidPredPerClassifier[[1]][[1]]))
    for (iClassifier in 1 : length(evidPredPerClassifier)){
      for (iModel in 1 : nModelPerBag){
        finalBF <- finalBF + evidPredPerClassifier[[iClassifier]][[iModel]]
      }
    }
    finalBF <- finalBF / length(evidPredPerClassifier)
  } else if (combinationType == "Dempster"){
    if (classifierName == "EBag1"){
      finalBF <- matrix(0, nrow = nrow(evidPredPerClassifier[[1]][[1]]), ncol = ncol(evidPredPerClassifier[[1]][[1]]))
      for (iClassifier in 1 : length(evidPredPerClassifier)){
        for (iModel in 1 : nModelPerBag){
          finalBF <- finalBF + evidPredPerClassifier[[iClassifier]][[iModel]]
        }
      }
      finalBF <- finalBF / (length(evidPredPerClassifier) * nModelPerBag)
    } else {
      finalBF <- evidPredPerClassifier[[1]][[1]]
      if (nModelPerBag > 1){
        for (iModel in 2 : nModelPerBag){
          m <- NULL
          for (i in 1 : nrow(finalBF)){
            m <- rbind(m, dempsterFusion(classifierName, finalBF[i, ], evidPredPerClassifier[[1]][[iModel]][i, ]))
          }
          finalBF <- m
        }
      }
      for (iClassifier in 2 : length(evidPredPerClassifier)){
        if (classifierName == "EBag1"){
          finalBF <- finalBF + evidPredPerClassifier[[iClassifier]][[iModel]]
        } else {
          for (iModel in 1 : nModelPerBag){
            m <- NULL
            for (i in 1 : nrow(finalBF)){
              m <- rbind(m, dempsterFusion(classifierName, finalBF[i, ], evidPredPerClassifier[[iClassifier]][[iModel]][i, ]))
            }
            finalBF <- m
          }
        }
      }
    }
  } else if (combinationType == "Disjunction"){
    finalBF <- evidPredPerClassifier[[1]][[1]]
    if (nModelPerBag > 1){
      for (iModel in 2 : nModelPerBag){
        m <- NULL
        for (i in 1 : nrow(finalBF)){
          m <- rbind(m, disjComb(classifierName, finalBF[i, ], evidPredPerClassifier[[1]][[iModel]][i, ]))
        }
        finalBF <- m
      }
    }
    
    for (iClassifier in 2 : length(evidPredPerClassifier)){
      for (iModel in 1 : nModelPerBag){
        m <- NULL
        for (i in 1 : nrow(finalBF)){
          m <- rbind(m, disjComb(classifierName, finalBF[i, ], evidPredPerClassifier[[iClassifier]][[iModel]][i, ]))
        }
        finalBF <- m
      }
    }
  }

  return(finalBF)
}
#############################################################################################
dempsterFusion <- function(classifierName, m1, m2){ 
  
  # Powerset preprocessing:
  focalElts <- extractFocElts(names(m1))
  
  # Conflict computation:
  kappa <- 0
  for (iFocElt1 in 1 : length(focalElts)){
    focElt1 <- focalElts[[iFocElt1]]
    for (iFocElt2 in 1 : length(focalElts)){
      focElt2 <- focalElts[[iFocElt2]]
      if (length(intersect(focElt1, focElt2)) == 0){
        kappa <- kappa + (m1[iFocElt1] * m2[iFocElt2])
      }
    }
  }
  
  # Combination:
  finalMass <- c()
  for (iFocElt in 1 : length(focalElts)){
    focElt <- focalElts[[iFocElt]]
    m      <- 0
    for (iFocElt1 in 1 : length(focalElts)){
      focElt1 <- focalElts[[iFocElt1]]
      for (iFocElt2 in 1 : length(focalElts)){
        focElt2 <- focalElts[[iFocElt2]]
        if (length(intersect(focElt1, focElt2)) == length(focElt)){
          if (prod(intersect(focElt1, focElt2) == focElt)){
            m <- m + (m1[iFocElt1] * m2[iFocElt2])
          }
        }
      }
    }
    finalMass <- c(finalMass, m)
  }
  names(finalMass) <- names(m1)
  if (kappa == 1){
    if (classifierName %in% c("EBag2", "EBag4")){
      finalMass                    <- finalMass - finalMass # TOTAL IGNORANCE in case of complete
      finalMass[length(finalMass)] <- 1                     # conflict between sources (kappa = 1)
    }
  } else {
    finalMass <- finalMass / (1 - kappa)
  }

  return(finalMass)
}
#############################################################################################
disjComb <- function(classifierName, m1, m2){ 
  
  # Powerset preprocessing:
  focalElts <- extractFocElts(names(m1))
  
  # Combination:
  finalMass <- c()
  for (iFocElt in 1 : length(focalElts)){
    focElt <- focalElts[[iFocElt]]
    m      <- 0
    for (iFocElt1 in 1 : length(focalElts)){
      focElt1 <- focalElts[[iFocElt1]]
      for (iFocElt2 in 1 : length(focalElts)){
        focElt2 <- focalElts[[iFocElt2]]
        if (length(union(focElt1, focElt2)) == length(focElt)){
          if (union(focElt1, focElt2) == focElt){
            m <- m + (m1[iFocElt1] * m2[iFocElt2])
          }
        }
      }
    }
    finalMass <- c(finalMass, m)
  }
  names(finalMass) <- names(m1)
  
  return(finalMass)
}
#############################################################################################
extractFocElts <- function(focElts){ 
  
  extractedFodElts <- list(NULL)
  for (iFocElt in 1 : length(focElts)){
    focElt <- focElts[iFocElt]
    if(substr(focElt, start = 1, stop = 1) != "{" & substr(focElt, start = 1, stop = 5) != "Omega"){ # singleton case
      extractedFodElts[[iFocElt]] <- focElt
    } else {
      if (substr(focElt, start = 1, stop = 1) == "{"){
        focElt             <- strsplit(focElt, split = ", ")[[1]] # non-singleton split
        preprocessedFocElt <- c()
        for (iSingleton in 1 : length(focElt)){
          singleton <- focElt[iSingleton]
          if (substr(singleton, start = 1, stop = 1) == "{"){
            singleton <- substr(singleton, start = 2, stop = nchar(singleton))
          }
          if (substr(singleton, start = nchar(singleton), stop = nchar(singleton)) == "}"){
            singleton <- substr(singleton, start = 1, stop = nchar(singleton) - 1)
          }
          preprocessedFocElt <- c(preprocessedFocElt, singleton)
        }
        extractedFodElts[[iFocElt]] <- preprocessedFocElt
      } else {
        extractedFodElts[[iFocElt]] <- setdiff(focElts, focElt)
      }
    }
  }
  
  return(extractedFodElts)
}
#############################################################################################
evidDecision <- function(classifierName, evidPred){ 
  
  if (classifierName == c("EBag1")){
    evidClassPred <- c()
    for (i in 1 : nrow(evidPred)){
      evidClassPred <- c(evidClassPred, colnames(evidPred)[which.is.max(evidPred[i, ])])
    }
  } else if (classifierName %in% c("EBag2", "EBag4")){
    evidClassPred <- c()
    for (i in 1 : nrow(evidPred)){
      evidClassPred <- c(evidClassPred, colnames(evidPred)[which.is.max(evidPred[i, 1 : eval(ncol(evidPred) - 1)])])
    }
  } else if (classifierName == "EBag3"){
    focalElts   <- colnames(evidPred)
    classLabels <- c()
    for (ifocElt in 1 : length(focalElts)){
      if (substr(focalElts[ifocElt], start = 1, stop = 1) != "{"){
        classLabels <- c(classLabels, focalElts[ifocElt])
      }
    }
    classLabels   <- classLabels[order(classLabels)]
    K             <- length(classLabels)
    evidClassPred <- c()
    for (i in 1 : nrow(evidPred)){
      # evidClassPred <- c(evidClassPred, colnames(evidPred)[which.is.max(evidPred[i, 1 : eval(ncol(evidPred) - K)])])
      # evidClassPred <- c(evidClassPred, classLabels[which.is.max(contourFunction(evidPred[i, ]))])
      evidClassPred <- c(evidClassPred, classLabels[which.is.max(BetP(evidPred[i, ]))])
    }
  }
  
  return(evidClassPred)
}
#############################################################################################
proba2BF <- function(classifierName, classProbPred, confusionMatrix){ 
  
  classLabels <- colnames(classProbPred)
  classLabels <- classLabels[order(classLabels)]
  K           <- length(classLabels)
  
  bigFocElts <- c()
  for (k in 1 : K){
    bigFocElts <- c(bigFocElts, paste0("{", paste(setdiff(classLabels, classLabels[k]), collapse=", "), "}"))
  }
  focalElts <- union(classLabels, bigFocElts)
  
  evidentialPredictions <- c()
  for (i in 1 : nrow(classProbPred)){
    proba                  <- classProbPred[i, ]
    mostProbableLabelIndex <- which.max(proba)
    mostProbableLabel      <- classLabels[mostProbableLabelIndex]
    unlikelyLabels         <- paste0("{", paste(setdiff(colnames(classProbPred), mostProbableLabel), collapse=", "), "}")
    
    if (sum(confusionMatrix[mostProbableLabelIndex, ]) > 0){
      probaConfusion         <- confusionMatrix[mostProbableLabelIndex, mostProbableLabelIndex]/sum(confusionMatrix[mostProbableLabelIndex, ])
    } else {
      probaConfusion         <- 0
    }
    
    mass                   <- c(as.numeric(proba) * probaConfusion, rep(0, length(bigFocElts)))
    mass[unlikelyLabels == focalElts] <- 1 - probaConfusion
    names(mass)            <- focalElts
    evidentialPredictions  <- rbind(evidentialPredictions, mass)
  }
 
  return(evidentialPredictions)
}
#############################################################################################
computeProbaPred <- function(singleClassifierName, singleClassifier, testingData, classLabels){ 
  if (singleClassifierName %in% c("tree", "forest", "c.nnet")){
    classProbPred             <- predict(singleClassifier, testingData, type = "prob")
  } else if (singleClassifierName %in% c("bayes", "nnet")){
    classProbPred <- predict(singleClassifier, testingData, type = "raw")
    if (prod(is.na(classProbPred))){  
      classPred         <- predict(singleClassifier, testingData)
      levels(classPred) <- classLabels
      classProbPred     <- matrix(0, nrow = length(classPred), ncol = length(classLabels))
      for (i in 1 : nrow(testingData)){
        classProbPred[i, which(classPred[i] == classLabels)] <- 1
      }
      colnames(classProbPred) <- classLabels
    }
    if (ncol(classProbPred) == 1){
      classPred         <- predict(singleClassifier, testingData, type = "class")
      levels(classPred) <- classLabels
      classProbPred     <- matrix(0, nrow = length(classPred), ncol = length(classLabels))
      for (i in 1 : nrow(testingData)){
        classProbPred[i, which(classPred[i] == classLabels)] <- 1
      }
      colnames(classProbPred) <- classLabels
    }
    if (ncol(classProbPred) < length(classLabels)){
      missingLabels <- setdiff(classLabels, colnames(classProbPred))
      classProbPred <- cbind(classProbPred, matrix(0, nrow = nrow(classProbPred), ncol = length(missingLabels)))
      colnames(classProbPred)[eval(ncol(classProbPred) - length(missingLabels) + 1) : ncol(classProbPred)] <- missingLabels
    } else {
      levels(classProbPred) <- classLabels
      classProbPred         <- class2probaPred(classProbPred, classLabels)
    }
  } else if (singleClassifierName == "SVM"){
    classProbPred <- attr(predict(singleClassifier, testingData, probability = T), "probabilities")
  } else if (singleClassifierName == "lda"){
    classProbPred <- as.matrix(predict(singleClassifier, testingData[, setdiff(names(testingData), "class")])$posterior)
  }
  if (ncol(classProbPred) < length(classLabels)){
    missingLabels <- setdiff(classLabels, colnames(classProbPred))
    classProbPred <- cbind(classProbPred, matrix(0, nrow = nrow(classProbPred), ncol = length(missingLabels)))
    colnames(classProbPred)[eval(ncol(classProbPred) - length(missingLabels) + 1) : ncol(classProbPred)] <- missingLabels
  }
  classProbPred <- classProbPred[ , order(colnames(classProbPred))]

  return(classProbPred)
}
#############################################################################################
contourFunction <- function(mass){ 
  
  # Powerset preprocessing:
  focalElts <- extractFocElts(names(mass))
  Omega     <- c()
  for (iFocElt in 1 : length(focalElts)){
    Omega <- union(Omega, focalElts[[iFocElt]])
  }
  Omega <- Omega[order(Omega)]
  
  contour <- c()
  for (iSingleton in 1 : length(Omega)){
    pl <- 0
    for (iFocElt in 1 : length(focalElts)){
      if (length(intersect(Omega[iSingleton], focalElts[iFocElt])) > 0){
        pl <- pl + mass[iFocElt]
      }
    }
    contour <- c(contour, pl)
  }
  
  return(contour)
}
#############################################################################################
BetP <- function(mass){ 
  
  # Powerset preprocessing:
  focalElts <- extractFocElts(names(mass))
  Omega     <- c()
  for (iFocElt in 1 : length(focalElts)){
    Omega <- union(Omega, focalElts[[iFocElt]])
  }
  Omega <- Omega[order(Omega)]
  
  BetProb <- c()
  for (iSingleton in 1 : length(Omega)){
    P <- 0
    for (iFocElt in 1 : length(focalElts)){
      if (length(intersect(Omega[iSingleton], focalElts[iFocElt])) > 0){
        P <- P + mass[iFocElt]/length(focalElts[iFocElt])
      }
    }
    BetProb <- c(BetProb, P)
  }
  
  return(BetProb)
}
############################################################################################# CVfoldsIndexes <- createFolds(dataset$class)
# createFoldsME <- function(vec, nFolds = 10){ 
#   vec     <- as.character(vec)
#   labels  <- unique(vec)
#   labels  <- labels[order(labels)]
#   K       <- length(labels)
#   indexes <- list(NULL)
#   for (iLabel in 1 : K){
#     oneLabelIndexes <- which(vec == labels[iLabel])
#     oneLabelIndexes <- oneLabelIndexes[sample(1 : length(oneLabelIndexes))]
#     N1label         <- length(oneLabelIndexes)
#     for (iFold in 1 : nFolds){
#       if (length(indexes) >= iFold){
#         indexes[[iFold]] <- c(indexes[[iFold]], oneLabelIndexes[1 : eval(floor(N1label/nFolds))])
#       } else {
#         indexes[[iFold]] <- oneLabelIndexes[1 : eval(floor(N1label/nFolds))]
#       }
#       oneLabelIndexes  <- oneLabelIndexes[eval(floor(N1label/nFolds) + 1) : length(oneLabelIndexes)]
#     }
#     if (length(oneLabelIndexes) > 0){
#       for (iFold in 1 : nFolds){
#         if (length(oneLabelIndexes) > 0){
#           indexes[[iFold]] <- c(indexes[[iFold]], oneLabelIndexes[1])
#           oneLabelIndexes  <- oneLabelIndexes[ - 1]
#         }
#       }
#     }
#   }
#   
#   return(indexes)
# }
############################################################################################# 
createFoldsME <- function (y, k = 10, list = TRUE, returnTrain = FALSE) 
{
  if (class(y)[1] == "Surv") 
    y <- y[, "time"]
  if (is.numeric(y)) {
    cuts <- floor(length(y)/k)
    if (cuts < 2) 
      cuts <- 2
    if (cuts > 5) 
      cuts <- 5
    breaks <- unique(quantile(y, probs = seq(0, 1, length = cuts)))
    y <- cut(y, breaks, include.lowest = TRUE)
  }
  if (k < length(y)) {
    y <- factor(as.character(y))
    numInClass <- table(y)
    foldVector <- vector(mode = "integer", length(y))
    for (i in 1:length(numInClass)) {
      min_reps <- numInClass[i]%/%k
      if (min_reps > 0) {
        spares <- numInClass[i]%%k
        seqVector <- rep(1:k, min_reps)
        if (spares > 0) 
          seqVector <- c(seqVector, sample(1:k, spares))
        foldVector[which(y == names(numInClass)[i])] <- sample(seqVector)
      }
      else {
        foldVector[which(y == names(numInClass)[i])] <- sample(1:k, 
                                                               size = numInClass[i])
      }
    }
  }
  else foldVector <- seq(along = y)
  if (list) {
    out <- split(seq(along = y), foldVector)
    names(out) <- paste("Fold", gsub(" ", "0", format(seq(along = out))), 
                        sep = "")
    if (returnTrain) 
      out <- lapply(out, function(data, y) y[-data], y = seq(along = y))
  }
  else out <- foldVector
  out
}



