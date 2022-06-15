setwd("C:/Users/squar/Documents/hydraulic_ABM/Output Files/")
library(openxlsx)
library(dplyr)
library(lattice)
library(munsell)

db <- read.xlsx("naive_wfh.xlsx")

db <- db %>%
  mutate(t = t / 3600 / 24)

db['day'] <- floor(db$t)

db_day <- data.frame(day = rep(seq(1, 89, 1), 4))
db_day['type'] <- rep(c(rep('Mean',89),rep('Max.', 89)),2)
db_day['wfh'] <- c(rep('wfh',(89*2)), rep('nowfh',(89*2)))


db_day_col <- db %>% group_by(day) %>%
  summarise(mean_wfh = mean(demand_wfh),
            max_wfh = max(demand_wfh),
            mean_x = mean(demand_x),
            max_x = max(demand_x))

db_day_col <- db_day_col[1:90,]

for (i in 1:4){
  for (j in 1:89){
    db_day[j+((i-1)*89),4] <- db_day_col[j,(i+1)]
  }
}

cols <- c("#000000", "#E69F00")

key_wfh <- list(title="Scenario",
                space="right",
                text=list(c('Base', 'WFH')),
                lines=list(lty=c(1,2), col=cols, lwd=2))

xyplot(mean_wfh~day|type, data=db_day, group=wfh,
       type='l', lty=c(1,2), col=cols, lwd=2,
       xlab = 'Time (days)', ylab = 'Demand (ML)',
       key = key_wfh)

plot(db$t, db$sum_i_perc, col=cols[1], type='l', lty=1, lwd=2,
     xlab="Time (day)", ylab="Percent Population",
     ylim=c(0,1))
lines(db$t, db$sum_I_no, lty = 1, lwd=2, col=cols[2])
lines(db$t, db$wfh, lty=2, lwd=2, col=cols[1])
legend('topleft', c('WFH Infected', 'Base Infected', 'WFH'),
       lty=c(1,1,2), col=c(cols[1], cols[2], cols[1]))
