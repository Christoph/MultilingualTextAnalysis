# Labour Market Frame Dictionary (fr_lab)
# Important: Validated and thus recommended to be used for text annotation of lemmatized English migration-related texts

library(data.table)
library(stringi)
library(dplyr)


# Instruction: Please replace "all_text_mt_lemma" with the name of your text column. And "sample_ID" with the name of your text ID column.
# A text is coded as labour market-related if at least 1 sentence contains a migration-related word and a labour market-related word.


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

# migration dictionary
migration <- c('(?:^|\\W)asyl', '(?:^|\\W)immigrant', '(?:^|\\W)immigrat', '(?:^|\\W)migrant', '(?:^|\\W)migrat', '(?:^|\\W)refugee', '(?:^|\\W)people\\W*on\\W*the\\W*run', 
               '(?:^|\\W)foreigner', '(?:^|\\W)foreign\\W*background',
               '(?:^|\\W)(paperless|undocumented)', '(?:^|\\W)sans\\W*pap(ie|e)r', '(?:^|\\W)guest\\W*worker', '(?:^|\\W)foreign\\W*worker', 
               '(?:^|\\W)emigra','(?:^|\\W)brain\\W*drain',
               '(?:^|\\W)free\\W*movement', '(?:^|\\W)freedom\\W*of\\W*movement', '(?:^|\\W)movement\\W*of\\W*person', 
               '(?:^|\\W)unaccompanied\\W*child', '(?:^|\\W)unaccompanied\\W*minor')

employ <- c('(?:^|\\W)employ', '(?:^|\\W)unemploy')
x_employ <- '(charit|board|volunt|aid|federal|minister|ministry|administration|advisor|detention|social|housing|accommodation|department|buro|office|secretary|policy|camp|detention center|rescue|congress|authority|ngo|smugg)'

hire <- c('(?:^|\\W)hir(e|ing)')
x_hire <- '(charit|board|volunt|aid|federal|minister|ministry|administration|advisor|detention|social|housing|accommodation|department|buro|office|secretary|policy|camp|detention center|rescue|congress|authority|ngo|smugg)'

job <- c('(?:^|\\W)job\\W', '(?:^|\\W)(permanent|pay|paid)\\Wposition')
x_job <- '(charit|board|volunt|aid|federal|minister|ministry|administration|advisor|detention|social|housing|accommodation|department|buro|office|secretary|policy|camp|detention center|rescue|congress|authority|ngo|smugg)'

labour <- '(?:^|\\W)(labor|labour)'
x_labour <- c('(?:^|\\W)part(y|ies|.s)', '(?:^|\\W)(peer|mp|pm|chair|leader|representatives|spokesman|the conservative|the tories)','(?:^|\\W)(labor|labour)ist' ,'(?:^|\\W)laboratory') 

occupation <- c('(?:^|\\W)occupation' , '(?:^|\\W)vocation')
x_occupation <- c('(?:^|\\W)palestine|palestinian|detention center|camp|housing|accommodation')

wage <- c('(?:^|\\W)wage', '(?:^|\\W)paycheck', '(?:^|\\W)salary', '(?:^|\\W)(pay|paid).{0,33}\\Wwork', '(?:^|\\W)(high|low)\\Wincome','(?:^|\\W)make\\W*(a|my|your|his|her|our|their)\\W*(living|live)', 
          '(?:^|\\W)remunerat', '(?:^|\\W)earn', '(?:^|\\W)(spanish|uk|english|british|romanian|polish|hungarian|swedish|german)\\Wunion')

work <- c('(find|get|look\\Wfor|want).{0,10}\\Wwork',  '(?:^|\\W)(migrant|emigrant|immigrant|foreigner|guest||alien|expat|temporary)\\W.{0,10}\\Wwork', '(?:^|\\W)(demand|supply)\\W.{0,10}\\Wwork')


work_prep_a <- c('(?:^|\\W)work\\W.{0,10}(as|at|for|with|in|from|during|until|near)', '(?:^|\\W)(as|at|for|with|in|from|during|until|near).{0,33}\\Wwork') 
x_work_prep <- c('(?:^|\\W)(charit|board|volunt|aid|federal|minister|ministry|administration|advisor|detention|social|housing|accommodation|department|buro|office|secretary|policy|camp|detention center|rescue|congress|authority|ngo|smugg)', "(?:^|\\W)work\\W.{0,10}with\\W.{0,10}(refugee|immigrant|migrant|asyl)")


legality_work <- c('(?:^|\\W)(illegal|legal|black|clandestine|irregular)\\Wwork', '(?:^|\\W)moonlighting','(?:^|\\W)work.black')


work_other <- c('(?:^|\\W)workplace', '(?:^|\\W)training\\Wplace','(?:^|\\W)domestic.{0,10}\\Wwork','(?:^|\\W)right\\Wto\\Wwork','(?:^|\\W)work.{0,10}\\W(ethic|life)','(?:^|\\W)work.{0,10}\\Wpermit',
                '(?:^|\\W)import\\Wof\\Wwork', '(?:^|\\W)(live|living)\\Wand\\W(work|working)')


worker <- c('(?:^|\\W)worker', '(?:^|\\W)workforce' ,'(?:^|\\W)working.class',
            '(?:^|\\W)work.of.foreign')#
x_worker <- '(?:^|\\W)(charit|board|volunt|aid|federal|minister|ministry|administration|advisor|detention|social|housing|accommodation|department|buro|office|secretary|policy|camp|detention center|rescue|congress|authority|ngo|smugg).{0,10}worker'

staff <- '(?:^|\\W)staff\\W' 
x_staff <- c('(?:^|\\W)(charit|board|federal|minister|ministry|administration|ngo|authority|academic|camp|centre|police|teach)', '(german|english|spanish|romanian|polish|hungarian)\\W(immigration|migration|emigration|asylum)\\Wstaff')#

teacher <- '(?:^|\\W)teacher' 
x_teacher <- c('(?:^|\\W)(german|english|spanish|romanian|polish|hungarian)language\\Wteacher', '(?:^|\\W)language\\Wteacher','(?:^|\\W)teacher.{0,10}for.{0,10}\\W(refugee|immigrant|migrant|asyl)', 
               '(?:^|\\W)(kid|child)\\Wof\\W(refugee|immigrant|migrant)', '(?:^|\\W)(refugee|immigrant|migrant|asyl).{0,5}\\W(child|kid|student)', '(?:^|\\W)deal\\Wwith\\W(refugee|immigrant|migrant|asyl)')

career <- '(?:^|\\W)career'

entrepreneur <- c('(?:^|\\W)entrepreneur', '(?:^|\\W)(company|industry)\\Wneed', '(?:^|\\W)foreign.{0,25}(business|company|shop)', '(?:^|\\W)business(man|woman)') 

skill <- c('(?:^|\\W)(high|low)\\Wskill','(?:^|\\W)(unskilled|unskill)', '(?:^|\\W)bring\\Win.{0,5}\\Wskills ') 

###########
## create dictionary
###########

dict_name <- c("migration",   "employ", "x_employ",  "hire", "x_hire", 
               "job", "x_job",  "labour",  "x_labour",   "occupation",  "x_occupation", "wage",
               "work",  "work_prep_a",  "x_work_prep",  "legality_work", "work_other",  "worker", "x_worker",
               "staff", "x_staff","teacher", "x_teacher"  ,"career", "entrepreneur", "skill" ) 


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

count_per_dict_sentence$employ_combi <- case_when(
  count_per_dict_sentence$employ >=1 & count_per_dict_sentence$x_employ >=1 & count_per_dict_sentence$migration >=1~ 0,
  count_per_dict_sentence$employ >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$employ_combi[is.na(count_per_dict_sentence$employ_combi)] <- 0 

count_per_dict_sentence$hire_combi <- case_when(
  count_per_dict_sentence$hire >=1 & count_per_dict_sentence$x_hire >=1 & count_per_dict_sentence$migration >=1~ 0,
  count_per_dict_sentence$hire >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$hire_combi[is.na(count_per_dict_sentence$hire_combi)] <- 0 

count_per_dict_sentence$job_combi <- case_when(
  count_per_dict_sentence$job >=1 & count_per_dict_sentence$x_job >=1 &count_per_dict_sentence$migration >=1~ 0,
  count_per_dict_sentence$job >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$job_combi[is.na(count_per_dict_sentence$job_combi)] <- 0 

count_per_dict_sentence$labour_combi <- case_when(
  count_per_dict_sentence$labour >=1 & count_per_dict_sentence$x_labour >=1& count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$labour >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$labour_combi[is.na(count_per_dict_sentence$labour_combi)] <- 0 

count_per_dict_sentence$occupation_combi <- case_when(
  count_per_dict_sentence$occupation >=1 & count_per_dict_sentence$x_occupation >=1& count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$occupation>=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$occupation_combi[is.na(count_per_dict_sentence$occupation_combi)] <- 0

count_per_dict_sentence$wage_combi <- case_when(
  count_per_dict_sentence$wage >=1  & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$wage_combi[is.na(count_per_dict_sentence$wage_combi)] <- 0 

count_per_dict_sentence$work_combi <- case_when(
  count_per_dict_sentence$work >=1& count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$work_combi[is.na(count_per_dict_sentence$work_combi)] <- 0 

count_per_dict_sentence$work_prep_a_combi <- case_when(
  count_per_dict_sentence$work_prep_a >=1 & count_per_dict_sentence$x_work_prep >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$work_prep_a >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$work_prep_a_combi[is.na(count_per_dict_sentence$work_prep_a_combi)] <- 0 

count_per_dict_sentence$legality_work_combi <- case_when(
  count_per_dict_sentence$legality_work >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$legality_work_combi[is.na(count_per_dict_sentence$legality_work_combi)] <- 0 

count_per_dict_sentence$work_other_combi <- case_when(
  count_per_dict_sentence$work_other >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$work_other_combi[is.na(count_per_dict_sentence$work_other_combi)] <- 0 

count_per_dict_sentence$worker_combi <- case_when(
  count_per_dict_sentence$worker >=1 & count_per_dict_sentence$x_worker >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$worker >=1& count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$worker_combi[is.na(count_per_dict_sentence$worker_combi)] <- 0 


count_per_dict_sentence$staff_combi <- case_when(
  count_per_dict_sentence$staff >=1 & count_per_dict_sentence$x_staff >=1& count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$staff >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$staff_combi[is.na(count_per_dict_sentence$staff_combi)] <- 0 

count_per_dict_sentence$teacher_combi <- case_when(
  count_per_dict_sentence$teacher >=1 & count_per_dict_sentence$x_teacher >=1 & count_per_dict_sentence$migration >=1~ 0,
  count_per_dict_sentence$teacher >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$teacher_combi[is.na(count_per_dict_sentence$teacher_combi)] <- 0 

count_per_dict_sentence$career_combi <- case_when(
  count_per_dict_sentence$career >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$career_combi[is.na(count_per_dict_sentence$career_combi)] <- 0 

count_per_dict_sentence$entrepreneur_combi <- case_when(
  count_per_dict_sentence$entrepreneur >=1 & count_per_dict_sentence$migration >=1~ 1)
count_per_dict_sentence$entrepreneur_combi[is.na(count_per_dict_sentence$entrepreneur_combi)] <- 0 

count_per_dict_sentence$skill_combi <- case_when(
  count_per_dict_sentence$skill >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$skill_combi[is.na(count_per_dict_sentence$skill_combi)] <- 0 


#check results on sentence level
#add sentence id for merging (order is correct)


count_per_dict_sentence$articleid_sent_id <- rownames(count_per_dict_sentence)
corpus_sentence$articleid_sent_id <- rownames(corpus_sentence)

#merge sentence and hits
corpus_sentence_with_dict_hits <- merge(corpus_sentence, count_per_dict_sentence, by = "articleid_sent_id")


#calc fr_lab variable
corpus_sentence_with_dict_hits$fr_lab <- case_when(corpus_sentence_with_dict_hits$employ_combi >=  1 | corpus_sentence_with_dict_hits$hire_combi >=  1 | 
                                                              corpus_sentence_with_dict_hits$job_combi >=  1 |corpus_sentence_with_dict_hits$labour_combi >=  1 |
                                                              corpus_sentence_with_dict_hits$occupation_combi >=  1 |   corpus_sentence_with_dict_hits$wage_combi >=  1 |
                                                              corpus_sentence_with_dict_hits$work_combi >= 1 | corpus_sentence_with_dict_hits$work_prep_a_combi >= 1 |   corpus_sentence_with_dict_hits$legality_work_combi >= 1 | corpus_sentence_with_dict_hits$work_other_combi >= 1 |corpus_sentence_with_dict_hits$worker_combi >= 1 | 
                                                              corpus_sentence_with_dict_hits$staff_combi  >= 1 | corpus_sentence_with_dict_hits$teacher_combi >= 1 | corpus_sentence_with_dict_hits$career_combi >= 1 | corpus_sentence_with_dict_hits$entrepreneur_combi >= 1 |  corpus_sentence_with_dict_hits$skill_combi >= 1   ~ 1)   


#aggregate the sentence level hit results on article level 


corpus_with_fr_agg <- corpus_sentence_with_dict_hits %>% 
  group_by(sample_ID) %>% 
  summarise_at(vars("fr_lab"), mean)


corpus_new <- merge(corpus_with_fr_agg, corpus, by = "sample_ID", all =T)


corpus_new <- subset(corpus_new, select = c(sample_ID, fr_lab))
corpus_new <- corpus_new %>% mutate(fr_wel = if_else(is.na(fr_lab), 0, fr_wel)) #set NA to 0 (necessary for sum within group later)

table(corpus_new$fr_lab) # descriptive overview



######
#save annotated dataset
###########
write.csv(corpus_new, "fr_lab.csv") # 




