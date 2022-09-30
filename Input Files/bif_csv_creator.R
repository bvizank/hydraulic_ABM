#### clear the environment ####

rm(list = ls())
cat("\014")

#### Change working directory ####
setwd("C:/Users/squar/OneDrive - North Carolina State University/Research/Code/BBN/country_breakdown")
# setwd("/Users/vizan/OneDrive - North Carolina State University/Research/Code/BBN/country_breakdown")

#### global variables ####
test_groups <- c('demo','media','trust','cultcog','personal','finite')
country_list <- c('IT', 'UK', 'US', 'AU', 'DE', 'ES', 'MX', 'SE', 'JP', 'KR', 'CN')
selection_proc = 'FS'       # either FS for forward selection or BE for backward elimination
survey_date = 1             # either 1 or 2 for march or april

rds_path <- paste0('./tmp_database.RDS')

# start a timer
ptm <- proc.time()

# Call all functions and install necessary packages ---------
# package installation
source(file = './package_installation.R')
# load all written functions
source(file = './functions.R')

# Read dataset -------------
db <- read.xlsx('./data_file_reduced.xlsx')
# remove descriptive rows
db <- db[-c(1:3),]

# predictors
db_x <- data.frame(lapply(db[,1:110], as.factor))
db_x <- subset(db_x, Survey_round == 1)
names(db_x)[names(db_x) == "Ethnic.min"] = "Ethnicmin"

setwd("C:/Users/squar/Documents/hydraulic_ABM/Input Files/")
dine <- read.bif('data_driven_dine.bif')
wfh <- read.bif('data_driven_wfh.bif')
grocery <- read.bif('data_driven_grocery.bif')

all_nodes <- c(nodes(dine), nodes(wfh), nodes(grocery))
all_nodes <- all_nodes[!all_nodes %in% c("dine_out_less", "work_from_home", "shop_groceries_less")]
all_nodes <- all_nodes[!duplicated(all_nodes)]

abm_out <- db_x[all_nodes]
abm_out <- cleaning(abm_out)
write.csv(abm_out, 'all_bbn_data.csv')
