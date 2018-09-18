#!/usr/bin/env Rscript

#source("https://bioconductor.org/biocLite.R")
#biocLite("stringi")
#biocLite("qvalue")
#install.packages("E:/masters/benchmark/virfinder/VirFinder/windows/VirFinder_1.1.zip", repos = NULL, type="source")  


library(VirFinder)

fna_dir <- file.path("E:/masters/data/3-fragments/fna")
output_dir <- file.path("E:/masters/benchmark/results")


run_virFinder <- function(testPath) {
  ptm <- proc.time()
  ## (2) prediction
  predResult <- VF.pred(testPath)
  #### (2.1) sort sequences by p-value in ascending order
  predResult <- predResult[order(predResult$pvalue),]
  
  out_path <- file.path(output_dir,paste(testPath, "csv", sep="."))
  write.csv(file=out_path, x=predResult)
  proc.time() - ptm
  
}

for(test_file in list.files(fna_dir)){
  if(grepl('test',test_file)){
    testPath  <- file.path(fna_dir,test_file)
    run_virFinder(testPath)

  }
}


  
