#!/usr/bin/env Rscript

#source("https://bioconductor.org/biocLite.R")
#biocLite("stringi")
#biocLite("qvalue")
#install.packages("E:/masters/benchmark/virfinder/VirFinder/windows/VirFinder_1.1.zip", repos = NULL, type="source")  


library(VirFinder)


## (1) train the model using user's database
#### (1.1) specifiy the fasta files of the training contigs, one for virus and one for prokaryotic hosts
trainFaFileHost <- file.path("../../data/2-train_test/non_viral_train.fna")
trainFaFileVirus <- file.path("../../data/2-train_test/viral_train.fna")

#### (1.2) specify the directory where the trained model will be saved to, and the name of the model
userModDir <- file.path('../../benchmark/model')
userModName <- "virnetDataModel"

## (2) train the model using user's database
w <- 8 # the length of the k-tuple word
VF.trainModUser <- VF.train.user(trainFaFileHost, trainFaFileVirus, userModDir, userModName, w, equalSize=TRUE)

## (3) load the trained model based on user's database
#modFile <- list.files(userModDir, userModName, full.names=TRUE)
#load(modFile)