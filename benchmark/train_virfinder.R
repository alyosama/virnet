#!/usr/bin/env Rscript

#source("https://bioconductor.org/biocLite.R")
#biocLite("stringi")
#biocLite("qvalue")
#install.packages("E:/masters/benchmark/virfinder/VirFinder/windows/VirFinder_1.1.zip", repos = NULL, type="source")  


library(VirFinder)


## (1) train the model using user's database
#### (1.1) specifiy the fasta files of the training contigs, one for virus and one for prokaryotic hosts
trainFaFileHost <- system.file("data", "host.fa", package="VirFinder")
trainFaFileVirus <- system.file("data", "virus.fa", package="VirFinder")

#### (1.2) specify the directory where the trained model will be saved to, and the name of the model
userModDir <- file.path(find.package("VirFinder"))
userModName <- "virFinderModel"

## (2) train the model using user's database
w <- 4 # the length of the k-tuple word
VF.trainModUser <- VF.train.user(trainFaFileHost, trainFaFileVirus, userModDir,
                                 userModName, w, equalSize=TRUE)
## (3) load the trained model based on user's database
modFile <- list.files(userModDir, userModName, full.names=TRUE)
load(modFile)