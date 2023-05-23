library(bnlearn)


setwd("C:\\Users\\squar\\Documents\\hydraulic_ABM\\Input Files\\pmt_models")

for (i in list.files()){
  if (i != 'net_files'){
    fitted <- read.bif(i)
    out_file <- paste0(substr(i, 1, nchar(i)-3), 'net')
    write.net(out_file, fitted)
  }
}