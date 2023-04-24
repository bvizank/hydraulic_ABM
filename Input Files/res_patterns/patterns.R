library(dplyr)

# setwd("C:\\Users\\bvizank\\Documents\\hydraulic_abm\\Input Files\\res_patterns")
setwd("/Users/vizan/OneDrive - North Carolina State University/Hourly_data_csv_2020")

mar_data <- read.csv("consumption_2020_03.csv")
apr_data <- read.csv("consumption_2020_04.csv")
may_data <- read.csv("consumption_2020_05.csv")

demo_data <- read.csv("~/Documents/hydraulic_abm/Input Files/res_patterns/miu_lot_merge_trunc_02.csv")

res_ids <- subset(demo_data, PropertyType == 'SFR')[['MiuId']]
res_ids <- paste0('X', res_ids)

missing_may <- setdiff(res_ids[-c(1)], colnames(may_data))

res_ids_may <- res_ids[!res_ids %in% missing_may]

mar_data <- mar_data[c('TimeStamp', res_ids_may)]
apr_data <- apr_data[c('TimeStamp', res_ids_may)]
may_data <- may_data[c('TimeStamp', res_ids_may)]

# total_data <- rbind(read.csv("consumption_2020_03.csv"), read.csv("consumption_2020_04.csv"))
total_data <- rbind(mar_data[529:nrow(mar_data),], apr_data, may_data)

daily_avgs <- list()
for (i in 0:7){
  week <- total_data[((i*168)+1):((i+1) * 168),]
  week <- sapply(week[,-c(1)], function(x) as.numeric(x))
  week <- week[ ,colSums(is.na(week)) == 0]
  daily_avgs[[i+1]] <- rowMeans(week)
}

hourly_avgs <- c()
hourly_norm <- c()
for (i in 1:length(daily_avgs)){
  days <- c()
  for (j in 0:6){
    week <- daily_avgs[[i]]
    days <- cbind(days, week[((j*24)+1):((j+1)*24)])
  }
  hourly_avgs <- cbind(hourly_avgs, rowMeans(days))
  hourly_norm <- cbind(hourly_norm, rowMeans(days)/mean(rowMeans(days)))
}

col_names <- c('w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'w7', 'w8')

hourly_avgs <- as.data.frame(hourly_avgs)
colnames(hourly_avgs) <- col_names

hourly_norm <- as.data.frame(hourly_norm)
colnames(hourly_norm) <- col_names

plot(hourly_avgs[[1]], type='l', ylim=c(0.5, 2.5),
     xlab="Time of Day [hr]", ylab="Total Demand [")
for (i in 2:length(hourly_avgs)){
  lines(hourly_avgs[[i]], type='l', col=i)
}
legend("top", legend=c(colnames(hourly_avgs)), col=1:8, lty=1, ncol=4)
dev.new()

plot(hourly_norm[[1]], type='l', ylim=c(0.5, 2.5))
for (i in 2:length(hourly_norm)){
  lines(hourly_norm[[i]], type='l', col=i)
}
legend("top", legend=c(colnames(hourly_norm)), col=1:8, lty=1, ncol=4)

write.csv(hourly_norms, '~/Documents/hydraulic_abm/Input Files/res_patterns/normalized_res_patterns.csv')
