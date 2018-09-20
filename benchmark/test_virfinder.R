#!/usr/bin/env Rscript

#source("https://bioconductor.org/biocLite.R")
#biocLite("stringi")
#biocLite("qvalue")
#install.packages("E:/masters/benchmark/virfinder/VirFinder/windows/VirFinder_1.1.zip", repos = NULL, type="source")  

#install.packages("doParallel") 

library(VirFinder)
library(foreach)
library(doParallel)

fna_dir <- file.path("../../data/3-fragments/fna")
output_dir <- file.path("../../benchmark/results")


run_virFinder <- function(testPath) {
  test_file <- basename(testPath)
  cat("Current working file: ", test_file,'\n')
  ## (2) prediction
  capture.output(predResult <- VF.pred(testPath), file='/dev/null')

  
  #### (2.1) sort sequences by p-value in ascending order
  predResult <- predResult[order(predResult$pvalue),]
  
  out_path <- file.path(output_dir,paste(test_file, "csv", sep="."))
  write.csv(file=out_path, x=predResult)  
  cat("Finished file: ", test_file,'\n')
}


no_cores <- detectCores() - 1
registerDoParallel(cores=no_cores)
getDoParWorkers()

run_fragments <- function(){
  print('Start VirFinder Fragments')
  fragments_files <- list.files(fna_dir)
  foreach(test_file = fragments_files) %dopar% {
    if(grepl('test',test_file)){
      testPath  <- file.path(fna_dir,test_file)
      system.time(run_virFinder(testPath))
    }
  }
}

run_metgenome<- function(){
  print('Start VirFinder Metagenome')
  fragments_files <- list('../../data/4-metagenome/microbiome/microbiome-reads.fa','../../data/4-metagenome/virome/virome-reads.fa')
  foreach(test_file = fragments_files) %dopar% {
    system.time(run_virFinder(test_file))
  }
}

#run_fragments()
run_metgenome()




  
