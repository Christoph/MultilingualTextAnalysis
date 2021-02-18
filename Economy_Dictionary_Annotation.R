# Economy & Budget Frame Dictionary (fr_eco)
# Validated and thus recommended to be used for text annotation of lemmatized English migration-related texts

library(data.table)
library(stringi)
library(dplyr)


# Instruction: Please replace "all_text_mt_lemma" with the name of your text column. And "sample_ID" with the name of your text ID column.
# A text is coded as economy and budget-related if at least 1 sentence contains a migration-related word and a economy and budget-related word.


#################
# read and prepare text data
##############

setwd('') #  directory 
corpus <- fread("") # your dataset

# lower case text column
corpus$all_text_mt_lemma <- tolower(corpus$all_text_mt_lemma)




###########
## dictionary components
###########

# migration-related keywords
migration <- c('(?:^|\\W)asyl', '(?:^|\\W)immigrant', '(?:^|\\W)immigrat', '(?:^|\\W)migrant', '(?:^|\\W)migrat', '(?:^|\\W)refugee', '(?:^|\\W)people\\W*on\\W*the\\W*run', 
               '(?:^|\\W)foreigner', '(?:^|\\W)foreign\\W*background',
               '(?:^|\\W)(paperless|undocumented)', '(?:^|\\W)sans\\W*pap(ie|e)r', '(?:^|\\W)guest\\W*worker', '(?:^|\\W)foreign\\W*worker', 
               '(?:^|\\W)emigra','(?:^|\\W)brain\\W*drain',
               '(?:^|\\W)free\\W*movement', '(?:^|\\W)freedom\\W*of\\W*movement', '(?:^|\\W)movement\\W*of\\W*person', 
               '(?:^|\\W)unaccompanied\\W*child', '(?:^|\\W)unaccompanied\\W*minor')


budget <-  '(?:^|\\W)budget'

cost <- '(?:^|\\W)cost\\W'
exclude_cost <- c('(?:^|\\W)human\\W*cost\\W', '(?:^|\\W)humanitarian\\W*cost\\W')

econom <- '(?:^|\\W)econom'

financ <- '(?:^|\\W)financ(?!ial\\stimes)'

fund <- '(?:^|\\W)(funding|funded|funds|fund)\\W' #not fundamentally
exclude_fund <- c('(?:^|\\W)charit', '(?:^|\\W)non\\W*governmental\\W*organi(s|z)ation', '(?:^|\\W)ngo\\s', '(?:^|\\W)donat', '(?:^|\\W)humanitarian\\W') 


gdp <- c('(?:^|\\W)gdp', '(?:^|\\W)gross\\W*domestic\\W*product', '(?:^|\\W)global\\W*domestic\\W*product','(?:^|\\W)gross\\W*national\\W*product')
exclude_gdp <- "(?:^|\\W)police\\W*union"

money <- c('(?:^|\\W)money', '(?:^|\\W)monetar', '(?:^|\\W)public\\W*money')



numbers <- c("(?:^|\\W)(eur|€|\\$|£|gbp|dm|kr|sek|ft|huf|zł|pln|l|rol|ron)\\W*[0-9]{1,6}\\W*(thousand|million|billion|mrd|m)\\W",              
             "(?:^|\\W)(eur|€|\\$|£|gbp|dm|kr|sek|ft|huf|zł|pln|l|rol|ron)\\W*[0-9]{0,6}.?[0-9]{0,3}\\W*(thousand|million|billion|mrd|m)\\W", 
             "(?:^|\\W)(eur|€|\\$|£|gbp|dm|kr|sek|ft|huf|zł|pln|l|rol|ron)\\W*[0-9]{0,6}\\W*(thousand|million|billion|mrd|m)\\W",               
             "(?:^|\\W)(thousand|million|billion|mrd|m)\\W*(euro|dollar|pound|mark|krona|kronor|peseta|zloty|forint|romanian\\Wleu|kronor)\\W",
             "(?:^|\\W)[0-9]{0,6}.?[0-9]{1,3}\\W*(euro|dollar|pound|mark|krona|kronor|peseta|zloty|forint|romanian\\Wleu|kronor)\\W")


tax <- '(?:^|\\W)tax(?!i)'

exclude_tax <- '(?:^|\\W)taxi'

other <- c('(?:^|\\W)government\\W*spend\\s*', '(?:^|\\W)resource', '(?:^|\\W)public\\W*spend', '(?:^|\\W)remittance')


exclude <- c("(?:^|\\W)traffick", "(?:^|\\W)smugg", "(?:^|\\W)arrest", "(?:^|\\W)fraud") #more genral crime/criminal/illegal does not really help

###########
## create dictionary
###########

dict_name <- c("migration", "budget", "cost", "exclude_cost", 
               "econom", "financ", "fund", "exclude_fund", "gdp",  
               "exclude_gdp", "money", "numbers","tax", "exclude_tax",
               "other", "exclude") 



###########
## sentence splitting 
###########

corpus$all_text_mt_lemma <- as.character(corpus$all_text_mt_lemma)

corpus$all_text_mt_lemma <- gsub("(www)\\.(.+?)\\.([a-z]+)", "\\1\\2\\3",corpus$all_text_mt_lemma) ## Punkte aus Webseiten entfernen: 

corpus$all_text_mt_lemma <- gsub("([a-zA-Z])\\.([a-zA-Z])", "\\1. \\2",corpus$all_text_mt_lemma) ## Verbleibende Punkte zwischen Buchstaben, Leerzeichen einfügen: 


corpus$all_text_mt_lemma <- as.character(corpus$all_text_mt_lemma)
list_sent <- strsplit(corpus$all_text_mt_lemma, "(!\\s|\\.\\s|\\?\\s)") 
corpus_sentence <- data.frame(sample_ID=rep(corpus$sample_ID, sapply(list_sent, length)), sentence=unlist(list_sent))


#######
## search pattern of dictionary in text corpus
#######


n <- length(dict_name) 
evaluation_sent <- vector("list", n) 

count <- 0


for (i in dict_name) {
  count <- count + 1
  print(count)
  print(i)
  match <- stri_count_regex(corpus_sentence$sentence, paste(get(i), collapse='|'))
  evaluation_sent[[i]] <- data.frame(name=match)
}


evaluation_sent <- evaluation_sent[-c(1:n)] 
count_per_dict_sentence <- do.call("cbind", evaluation_sent) 

cols <- names(count_per_dict_sentence) == "name" 
names(count_per_dict_sentence)[cols] <- paste0("name", seq.int(sum(cols))) 
oldnames <- colnames(count_per_dict_sentence) 
newnames <- names(evaluation_sent) 


setnames(count_per_dict_sentence, old = oldnames, new = newnames) 
head(count_per_dict_sentence)
colnames(count_per_dict_sentence)


###########
# some recoding
###########

count_per_dict_sentence$budget_combi <- case_when(
  count_per_dict_sentence$budget >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$budget_combi[is.na(count_per_dict_sentence$budget_combi)] <- 0 

count_per_dict_sentence$cost_combi <- case_when(
  count_per_dict_sentence$cost >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude_cost ==1~ 0,
  count_per_dict_sentence$cost >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude >=1~ 0,
  count_per_dict_sentence$cost >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$cost_combi[is.na(count_per_dict_sentence$cost_combi)] <- 0 

count_per_dict_sentence$econom_combi <- case_when(
  count_per_dict_sentence$econom >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$econom_combi[is.na(count_per_dict_sentence$econom_combi)] <- 0 

count_per_dict_sentence$financ_combi <- case_when(
  count_per_dict_sentence$financ >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude >=1~ 0,
  count_per_dict_sentence$financ >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$financ_combi[is.na(count_per_dict_sentence$financ_combi)] <- 0 

count_per_dict_sentence$fund_combi <- case_when(
  count_per_dict_sentence$fund >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude_fund >=1~ 0,
  count_per_dict_sentence$fund >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$fund_combi[is.na(count_per_dict_sentence$fund_combi)] <- 0 

count_per_dict_sentence$gdp_combi <- case_when(
  count_per_dict_sentence$gdp >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude_gdp >=1~ 0,
  count_per_dict_sentence$gdp >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$gdp_combi[is.na(count_per_dict_sentence$gdp_combi)] <- 0 

count_per_dict_sentence$money_combi <- case_when(
  count_per_dict_sentence$money >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude >=1~ 0,
  count_per_dict_sentence$money >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude_fund >=1~ 0,
  count_per_dict_sentence$money >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$money_combi[is.na(count_per_dict_sentence$money_combi)] <- 0 

count_per_dict_sentence$numbers_combi <- case_when(
  count_per_dict_sentence$numbers >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude >=1~ 0,
  count_per_dict_sentence$numbers >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$numbers_combi[is.na(count_per_dict_sentence$numbers_combi)] <- 0 

count_per_dict_sentence$tax_combi <- case_when(
  count_per_dict_sentence$tax >=1 & count_per_dict_sentence$migration >=1 & count_per_dict_sentence$exclude_tax >=1~ 0,
  count_per_dict_sentence$tax >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$tax_combi[is.na(count_per_dict_sentence$tax_combi)] <- 0 

count_per_dict_sentence$other_combi <- case_when(
  count_per_dict_sentence$other >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$other_combi[is.na(count_per_dict_sentence$other_combi)] <- 0 


#check results on sentence level
# add sentence id for merging (order is correct)


count_per_dict_sentence$reminderid_sent_id <- rownames(count_per_dict_sentence)
corpus_sentence$reminderid_sent_id <- rownames(corpus_sentence)

#merge sentence and hits
corpus_sentence_with_dict_hits <- merge(corpus_sentence, count_per_dict_sentence, by = "reminderid_sent_id")


#aggregate the sentence level hit results on article level (sum up results of several variables per article (doc_id))#all those that were searched for on sentence level

df_sent_results_agg <- corpus_sentence_with_dict_hits %>% 
  group_by(sample_ID) %>% 
  summarise_at(vars("migration", "budget_combi", "cost_combi", "exclude_cost", "econom_combi", "financ_combi", "fund_combi", "gdp_combi", "money_combi", "numbers_combi", "tax_combi", "other_combi", "exclude"), sum)



corpus_new <- merge(df_sent_results_agg, corpus, by = "sample_ID", all =T)


#recode combination of dictionary components that make up the dictionary



#calc fr_eco variable
corpus_sentence_with_dict_hits$fr_eco <- case_when(corpus_sentence_with_dict_hits$budget_combi >= 1 | corpus_sentence_with_dict_hits$cost_combi >= 1 | corpus_sentence_with_dict_hits$econom_combi >= 1 |
                                                              corpus_sentence_with_dict_hits$financ_combi >= 1 | corpus_sentence_with_dict_hits$fund_combi >= 1 | corpus_sentence_with_dict_hits$gdp_combi >= 1 |
                                                              corpus_sentence_with_dict_hits$numbers_combi >= 1 | corpus_sentence_with_dict_hits$money_combi >= 1 |corpus_sentence_with_dict_hits$tax_combi >= 1 | corpus_sentence_with_dict_hits$other_combi >= 1 ~ 1)#


#aggregate the sentence level hit results on article level 


corpus_with_fr_agg <- corpus_sentence_with_dict_hits %>% 
  group_by(sample_ID) %>% 
  summarise_at(vars("fr_eco"), mean)


corpus_new <- merge(corpus_with_fr_agg, corpus, by = "sample_ID", all =T)


corpus_new <- subset(corpus_new, select = c(sample_ID, fr_eco))
corpus_new <- corpus_new %>% mutate(fr_eco = if_else(is.na(fr_eco), 0, fr_eco)) #set NA to 0 (necessary for sum within group later)

table(corpus_new$fr_eco) # descriptive overview



######
#save annotated dataset
###########
write.csv(corpus_new, "fr_eco.csv") # 



