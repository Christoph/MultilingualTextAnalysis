# Welfare Frame Dictionary (fr_wel)
# Validated and thus recommended to be used for text annotation of lemmatized (R package udpipe) English migration-related texts

library(data.table)
library(stringi)
library(dplyr)


# Instruction: Please replace "all_text_mt_lemma" with the name of your text column. And "sample_ID" with the name of your text ID column.
# A text is coded as welfare-related if at least 1 sentence contains a migration-related word and a welfare-related word.


#################
# read and prepare text data
##############

setwd('') #  directory 
corpus <- fread("") # your dataset with lemmatized text

# lowercase text column
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

# welfare-related keywords
welfare_general <- c('(?:^|\\W)welfare',
                     '(?:^|\\W)basic.{0,5}\\W(allowance|assistance|benefit|bonus|care|guarantee|insurance|package|protection|service|safety|security|support)',
                     '(?:^|\\W)(allowance|assistance|benefit|bonus|care|guarantee|insurance|package|protection|service|safety|security|support).{0,5}\\Wbasic',
                     '(?:^|\\W)(claim|cut|guarantee).{0,5}\\W(allowance|assistance|benefit|care|service|support)',
                     '(?:^|\\W)(allowance|assistance|benefit|care|service|support).{0,5}\\W(claim|cut|guarantee)',
                     '(?:^|\\W)social.{0,5}\\W(allowance|assistance|benefit|bonus|card|care|centre|guarantee|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system)', 
                     '(?:^|\\W)(allowance|assistance|benefit|bonus|card|care|centre|guarantee|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system).{0,5}\\Wsocial', 
                     '(?:^|\\W)public.{0,5}\\W(allowance|assistance|benefit|bonus|finance|insurance|money|package|payment|program|protection|resource|service|spend|system)', 
                     '(?:^|\\W)(allowance|assistance|benefit|bonus|finance|insurance|money|package|payment|program|protection|resource|service|spend|system).{0,5}\\Wpublic', 
                     '(?:^|\\W)(access|abuse).{0,5}\\Wbenefit', 
                     '(?:^|\\W)benefit.{0,5}\\W(access|abuse)', 
                     '(?:^|\\W)sachleistung', '(?:^|\\W)(monthly|weekly|daily)\\Wallowance', '(?:^|\\W)public\\Wtransport\\Wticket',
                     '(?:^|\\W)\\W*in\\W*kind\\W*(allowance|assistance|benefit|care|package|payment|program|protection|resource|service|spend|support|system)',
                     '(?:^|\\W)(allowance|assistance|benefit|care|package|payment|program|protection|resource|service|spend|support|system)\\W*in\\W*kind')




#education

education <- c('(?:^|\\W)educat')

kids <- c('(?:^|\\W)kindergarten',  '(?:^|\\W)childcare')

school <- c('(?:^|\\W)school', '(?:^|\\W)(integration|preparatory)\\W(course|class)','(?:^|\\W)teacher', '(?:^|\\W)highschool', '(?:^|\\W)pupil')




university <- c('(?:^|\\W)apprenticeship','(?:^|\\W)diploma\\W','(?:^|\\W)student', '(?:^|\\W)work\\Wand\\Wstudy', '(?:^|\\W)studi', #study does not work, the noun "study" "studies" is mostly unrealted, training to many false positves
                '(?:^|\\W)tuition','(?:^|\\W)universit', '(?:^|\\W)undergraduate', '(?:^|\\W)scholarship\\Wfor\\W(immigrant|migrant|refugee|asylum)',
                '(?:^|\\W)study\\W*support', '(?:^|\\W)alumn(a|us)')
university_exclude <- c('(?:^|\\W)migration\\Wexpert(s?)\\W', '(?:^|\\W)analyst\\W(at|of|from)\\W','(?:^|\\W)expert\\W(at|of|from)\\W',
                        '(?:^|\\W)study\\W(from|of|by)\\Wthe\\W', 
                        '(?:^|\\W)survey\\W(from|of|by)\\Wthe\\W', 
                        '(?:^|\\W)professor\\W(at|of|from)\\W', '(?:^|\\W)director\\W(at|of|from)\\W', '(?:^|\\W)scientist\\W(at|of|from)\\W',
                        '(?:^|\\W)lecturer\\W(at|of|from)\\W', '(?:^|\\W)rector\\W(at|of|from)\\W')




#child/family 

child_benefit <- c('(?:^|\\W)child.{0,5}\\Wrelate.{0,5}\\W(allowance|assistance|benefit|bonus|care|finance|guarantee|insurance|money|package|payment|program|promotion|protection|resource|service|spend|support|system)', 
                   '(?:^|\\W)child.{0,5}\\Wbasic\\W*(allowance|assistance|benefit|bonus|care|finance|guarantee|insurance|money|package|payment|program|promotion|protection|resource|security|service|spend|support|system)',
                   '(?:^|\\W)child.{0,5}\\W(allowance|assistance|benefit|bonus|care|finance|guarantee|insurance|money|package|payment|program|promotion|protection|resource|service|spend|support|system)',
                   '(?:^|\\W)(allowance|assistance|benefit|bonus|care|finance|guarantee|insurance|money|package|payment|program|promotion|protection|resource|service|spend).{0,5}\\Wchild')


family_benefit <- c('(?:^|\\W)family\\W*help\\W*(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system)',
                    '(?:^|\\W)family\\W*(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system)', 
                    '(?:^|\\W)(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system).{0,5}\\Wfamily*', 
                    '(?:^|\\W)mater(nity|nal).{0,5}\\W(allowance|benefit|bonus|care|money|payment|resource|spend)',
                    '(?:^|\\W)paren.{0,7}\\W(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system)',
                    '(?:^|\\W)(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system).{0,5}\\Wparen',
                    '(?:^|\\W)paternity.{0,7}\\W(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system)',
                    '(?:^|\\W)(allowance|benefit|bonus|care|centre|check|finance|guarantee|insurance|money|package|payment|program|protection|resource|service|spend|support|system).{0,5}\\Wpaternity')

#work

work <- c('(?:^|\\W)unemployment\\W*(allowance|assistance|benefit|card|care|centre|finance|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system)', '(?:^|\\W)in.?work\\Wbenefit',
          '(?:^|\\W)(allowance|assistance|benefit|card|care|centre|finance|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system).{0,5}\\Wunemployment', 
          '(?:^|\\W)employment\\W*(allowance|assistance|benefit|bonus|card|care|centre|insurance|item|money|package|payment|program|resource|service|safety|security|spend|support|system)',
          '(?:^|\\W)(allowance|assistance|benefit|bonus|card|care|centre|insurance|item|money|package|payment|program|resource|service|safety|security|spend|support|system).{0,5}\\Wemployment')




#housing

housing <- c('(?:^|\\W)(affordable|alternative|decent|finance|immediate|more|new|official|public|provide|reasonable|social|set\\Wup|setting\\sup).{0,5}\\W(housing|accommodation|tent\\Wcamp|tented\\Wcamp)', '(?:^|\\W)social\\W*rent', 
             '(?:^|\\W)(housing|accommodation).{0,5}\\W(allowance|assistance|benefit|bonus|guarantee|program|resource|shortage|support|stock)',
             '(?:^|\\W)housing.?price', '(?:^|\\W)rehous')


migrant_housing <- c('(?:^|\\W)(ensur|build|construct|contribut|creat|look\\Wfor|maintain|invest\\W|need|open|operat|requir|establish|responsible|start).{0,15}\\W(immigrant|migrant|refugee|asylum).{0,15}\\W(accommodation|center|dormitory|settlement|home|housing|shelter|reception\\scenter|Erstaufnahmestelle|camp)',
                     '(?:^|\\W)(ensur|build|construct|contribut|creat|look\\Wfor|maintain|invest\\W|need|open|operat|requir|establish|responsible|start).{0,15}\\W(accommodation|center|dormitory|settlement|home|housing|shelter|reception\\scenter|Erstaufnahmestelle|camp).{0,15}\\W(immigrant|migrant|refugee|asylum)',
                     "(?:^|\\W)(immigrant|migrant|refugee|asylum).{0,15}\\W(accommodation|center|dormitory|settlement|home|housing|shelter|reception\\scenter|Erstaufnahmestelle|camp).{0,15}\\W(ensur|build|construct|contribut|creat|look\\Wfor|maintain|invest\\W|need|open|operat|requir|establish|responsible|start)",
                     "(?:^|\\W)(accommodation|center|dormitory|settlement|home|housing|shelter|reception\\scenter|erstaufnahmestelle|camp).{0,15}\\W(immigrant|migrant|refugee|asylum).{0,15}\\W(ensur|build|construct|contribut|creat|look\\Wfor|maintain|invest\\W|need|open|operat|requir|establish|responsible|start)",
                     "(?:^|\\W)(accommodation|center|dormitory|settlement|home|housing|shelter|reception\\scenter|erstaufnahmestelle|camp).{0,15}\\W(ensur|build|construct|contribut|creat|look\\Wfor|maintain|invest\\W|need|open|operat|requir|establish|responsible|start).{0,15}\\Wfor\\W(immigrant|migrant|refugee|asylum)")

migrant_housing_exclude <- c('(?:^|\\W)smuggler', '(?:^|\\W)crime', '(?:^|\\W)fire', "(?:^|\\W)attack")




#food

food <- c('(?:^|\\W)food\\W*(bank|distribut|subsidy|voucher)', '(?:^|\\W)(distribut|provide|give|receive|serve).{0,5}\\W(food|water)')



#pension/retirement 

pension <- c('(?:^|\\W)pension\\W',
             '(?:^|\\W)retirement.{0,5}(allowance|assistance|benefit|bonus|care|guarantee|finance|insurance|money|package|payment|program|resource|service|safety|security|spend|support|system)',
             '(?:^|\\W)(allowance|assistance|benefit|bonus|care|guarantee|finance|insurance|money|package|payment|program|resource|service|safety|security|spend|support|system).{0,5}\\Wretirement',
             '(?:^|\\W)elderly.{0,5}(allowance|assistance|benefit|bonus|care|finance|insurance|money|package|payment|program|resource|service|safety|security|spend|support)',
             '(?:^|\\W)(allowance|assistance|benefit|bonus|care|finance|insurance|money|package|payment|program|resource|service|safety|security|spend|support).{0,5}\\Welderly')



#health/medical services


medi <- c('(?:^|\\W)medic(a|i)', 
          '(?:^|\\W)healthcare', 
          '(?:^|\\W)free\\W*health\\W*check', 
          '(?:^|\\W)health\\W*(allowance|assistance|benefit|bonus|card|care|centre|check|finance|guarantee|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system)',
          '(?:^|\\W)(allowance|assistance|benefit|bonus|card|care|centre|check|finance|guarantee|insurance|items|money|package|payment|program|protection|resource|service|safety|security|spend|support|system).{0,5}\\Whealth')	


care <- c('(?:^|\\W)care\\W*(allowance|assistance|benefit|bonus|card|care|centre|check|finance|guarantee|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system)', 
          '(?:^|\\W)(allowance|assistance|benefit|bonus|card|care|centre|check|finance|guarantee|insurance|item|money|package|payment|program|protection|resource|service|safety|security|spend|support|system).{0,5}\\Wcare',
          '(?:^|\\W)(basic|better|hospital|primary|provide|specialist|special)\\W*care')


medi_specific <- c( '(?:^|\\W)alcoholism', '(?:^|\\W)anaesthesiolog', 
                    '(?:^|\\W)cancer','(?:^|\\W)cardiolog', '(?:^|\\W)cardiothoracic','(?:^|\\W)cardiovascular', 
                    '(?:^|\\W)dental', '(?:^|\\W)dentist',
                    '(?:^|\\W)dermatolog', '(?:^|\\W)disabilit(y|ies)','(?:^|\\W)disable(d|ment)', '(?:^|\\W)handicap', '(?:^|\\W)with\\W*special\\W*needs',
                    '(?:^|\\W)endocrin','(?:^|\\W)gastroenterolog', '(?:^|\\W)geriatric',
                    '(?:^|\\W)gerontolog', '(?:^|\\W)gynaecolog','(?:^|\\W)haematolog', 
                    '(?:^|\\W)hematolog', 
                    '(?:^|\\W)hepatolog', '(?:^|\\W)hiv\\S',
                    '(?:^|\\W)immunis','(?:^|\\W)immuniz','(?:^|\\W)immunolog', '(?:^|\\W)influenza', '(?:^|\\W)hypothermia',
                    '(?:^|\\W)maxillofacial','(?:^|\\W)nephrolog', '(?:^|\\W)neurolog', '(?:^|\\W)neurosurger',  
                    '(?:^|\\W)obese',  '(?:^|\\W)obesity','(?:^|\\W)obstetric', '(?:^|\\W)oncolog',	'(?:^|\\W)opthalmolog', '(?:^|\\W)orthopaedic', '(?:^|\\W)orthopedic',
                    '(?:^|\\W)palliative', '(?:^|\\W)pandemic', '(?:^|\\W)patholog', '(?:^|\\W)pediatric',
                    '(?:^|\\W)proctolog','(?:^|\\W)physician', '(?:^|\\W)psychiatr','(?:^|\\W)psychological',
                    '(?:^|\\W)pulmonol',  '(?:^|\\W)radiolog',  '(?:^|\\W)radiotherap', '(?:^|\\W)rheumatolog', 
                    '(?:^|\\W)surger', '(?:^|\\W)surgical', '(?:^|\\W)treatment\\W*guarantee', '(?:^|\\W)tuberculosis',
                    '(?:^|\\W)urolog', '(?:^|\\W)vascular')	


hospi <- c('(?:^|\\W)ambulance','(?:^|\\W)go\\W*to\\W*the\\W*doctor', '(?:^|\\W)emergency\\W*room',
           '(?:^|\\W)hospital\\W', '(?:^|\\W)pharmac', '(?:^|\\W)doctor\\Wvisit',
           '(?:^|\\W)vaccin', '(?:^|\\W)treatment\\Wfor\\W(immigrant|migrant|refugee|asylum)')

sick <- c('(?:^|\\W)illness','(?:^|\\W)infectious','(?:^|\\W)sick')


###########
## create dictionary
###########

dict_name <- c("migration", 
               "welfare_general", 
               "education", "kids", "school",  "university", "university_exclude",
               "child_benefit", "family_benefit", 
               "work",  "housing", "migrant_housing", 'migrant_housing_exclude', "food", 
               "pension",  
               "medi", "care", "medi_specific", "hospi", "sick") 

###########
## sentense splitting 
###########

corpus$all_text_mt_lemma <- as.character(corpus$all_text_mt_lemma)

corpus$all_text_mt_lemma <- gsub("(www)\\.(.+?)\\.([a-z]+)", "\\1\\2\\3",corpus$all_text_mt_lemma) 

corpus$all_text_mt_lemma <- gsub("([a-zA-Z])\\.([a-zA-Z])", "\\1. \\2",corpus$all_text_mt_lemma) 


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

count_per_dict_sentence$welfare_general_combi <- case_when(
  count_per_dict_sentence$welfare_general >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$welfare_general_combi[is.na(count_per_dict_sentence$welfare_general_combi)] <- 0 

count_per_dict_sentence$education_combi <- case_when(
  count_per_dict_sentence$education >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$education_combi[is.na(count_per_dict_sentence$education_combi)] <- 0 

count_per_dict_sentence$kids_combi <- case_when(
  count_per_dict_sentence$kids >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$kids_combi[is.na(count_per_dict_sentence$kids_combi)] <- 0 

count_per_dict_sentence$school_combi <- case_when(
  count_per_dict_sentence$school >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$school_combi[is.na(count_per_dict_sentence$school_combi)] <- 0 

count_per_dict_sentence$university_combi <- case_when(
  count_per_dict_sentence$university_exclude >=1 & count_per_dict_sentence$university >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$university >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$university_combi[is.na(count_per_dict_sentence$university_combi)] <- 0 

count_per_dict_sentence$child_benefit_combi <- case_when(
  count_per_dict_sentence$child_benefit >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$child_benefit_combi[is.na(count_per_dict_sentence$child_benefit_combi)] <- 0 

count_per_dict_sentence$family_benefit_combi <- case_when(
  count_per_dict_sentence$family_benefit >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$family_benefit_combi[is.na(count_per_dict_sentence$family_benefit_combi)] <- 0 

count_per_dict_sentence$work_combi <- case_when(
  count_per_dict_sentence$work >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$work_combi[is.na(count_per_dict_sentence$work_combi)] <- 0 

count_per_dict_sentence$housing_combi <- case_when(
  count_per_dict_sentence$housing >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$housing_combi[is.na(count_per_dict_sentence$housing_combi)] <- 0 

count_per_dict_sentence$migrant_housing_combi <- case_when(
  count_per_dict_sentence$migrant_housing_exclude >=1 & count_per_dict_sentence$migrant_housing >=1 ~ 0,
  count_per_dict_sentence$migrant_housing >=1  ~ 1)
count_per_dict_sentence$migrant_housing_combi[is.na(count_per_dict_sentence$migrant_housing_combi)] <- 0 

count_per_dict_sentence$food_combi <- case_when(
  count_per_dict_sentence$food >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$food_combi[is.na(count_per_dict_sentence$food_combi)] <- 0 

count_per_dict_sentence$pension_combi <- case_when(
  count_per_dict_sentence$pension >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$pension_combi[is.na(count_per_dict_sentence$pension_combi)] <- 0 

count_per_dict_sentence$medi_combi <- case_when(
  count_per_dict_sentence$medi >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$medi_combi[is.na(count_per_dict_sentence$medi_combi)] <- 0 

count_per_dict_sentence$care_combi <- case_when(
  count_per_dict_sentence$care >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$care_combi[is.na(count_per_dict_sentence$care_combi)] <- 0 

count_per_dict_sentence$medi_specific_combi <- case_when(
  count_per_dict_sentence$medi_specific >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$medi_specific_combi[is.na(count_per_dict_sentence$medi_specific_combi)] <- 0 

count_per_dict_sentence$hospi_combi <- case_when(
  count_per_dict_sentence$hospi >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$hospi_combi[is.na(count_per_dict_sentence$hospi_combi)] <- 0 

count_per_dict_sentence$sick_combi <- case_when(
  count_per_dict_sentence$sick >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$sick_combi[is.na(count_per_dict_sentence$sick_combi)] <- 0 



#add sentence id for merging (order is correct)
count_per_dict_sentence$articleid_sent_id <- rownames(count_per_dict_sentence)
corpus_sentence$articleid_sent_id <- rownames(corpus_sentence)

#merge sentences and hits
corpus_sentence_with_dict_hits <- merge(corpus_sentence, count_per_dict_sentence, by = "articleid_sent_id")


#calc fr_wel variable
corpus_sentence_with_dict_hits$fr_wel <- case_when(corpus_sentence_with_dict_hits$welfare_general_combi >= 1| 
                                                              corpus_sentence_with_dict_hits$education_combi >= 1| corpus_sentence_with_dict_hits$kids_combi >= 1| corpus_sentence_with_dict_hits$school_combi >= 1|  corpus_sentence_with_dict_hits$university_combi >= 1|
                                                              corpus_sentence_with_dict_hits$child_benefit_combi >= 1| corpus_sentence_with_dict_hits$family_benefit_combi >= 1| corpus_sentence_with_dict_hits$work_combi >= 1|
                                                              corpus_sentence_with_dict_hits$housing >= 1| corpus_sentence_with_dict_hits$migrant_housing >= 1|corpus_sentence_with_dict_hits$food_combi >= 1|
                                                              corpus_sentence_with_dict_hits$pension_combi >= 1|
                                                              corpus_sentence_with_dict_hits$care_combi >= 1 | corpus_sentence_with_dict_hits$medi_combi >= 1 | corpus_sentence_with_dict_hits$medi_specific_combi >= 1 | corpus_sentence_with_dict_hits$hospi_combi >= 1 | corpus_sentence_with_dict_hits$sick_combi >= 1~ 1)#


#aggregate the sentence level hit results on article level 

corpus_with_fr_agg <- corpus_sentence_with_dict_hits %>% 
  group_by(sample_ID) %>% 
  summarise_at(vars("fr_wel"), mean)


corpus_new <- merge(corpus_with_fr_agg, corpus, by = "sample_ID", all =T)


corpus_new <- subset(corpus_new, select = c(sample_ID, fr_wel))
corpus_new <- corpus_new %>% mutate(fr_wel = if_else(is.na(fr_wel), 0, fr_wel)) #set NA to 0 

table(corpus_new$fr_wel) # descriptive overview


######
#save annotated dataset
###########
write.csv(corpus_new, "fr_wel.csv")  


