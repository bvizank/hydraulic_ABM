setwd("C:\\Users\\bvizank\\Documents\\hydraulic_abm\\Input Files\\res_patterns")

pattern1 <- read.csv("week20r.csv")

hourly <- rowSums(pattern1[,-c(1)])
