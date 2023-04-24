library(openxlsx)
library(dplyr)

setwd("C:\\Users\\squar\\Documents\\hydraulic_ABM\\Output Files\\2022-08-31_08-36_results")

demand <- read.xlsx('datasheet.xlsx', sheet='demand')
demand <- subset(demand, select=-c(Total, Tank, SurfaceResrvr, Aquifer))

row_means <- rowSums(subset(demand, select=-c(1)))
row_means <- as.data.frame(cbind(time=demand[[1]], day=rep(0:89, each=24), demand=row_means))

daily_max <- row_means %>%
  group_by(day) %>%
  summarise(max = max(demand), mean = mean(demand), max_time = which.max(demand))
