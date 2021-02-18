# Security Frame Dictionary (fr_sec)
# Important: Validated and thus recommended to be used for text annotation of lemmatized English migration-related texts

library(data.table)
library(stringi)
library(dplyr)


# Instruction: Please replace "all_text_mt_lemma" with the name of your text column. And "sample_ID" with the name of your text ID column.
# A text is coded as security-related if at least 1 sentence contains a migration-related word and a security-related word.



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



border <- c('(?:^|\\W)(barrier|border|fence|frontier|wall).{0,5}\\W(agency|authority|guard|patrol|troop)',
            '(?:^|\\W)(barrier|border|fence|wall).{0,5}\\W(build|construct|set\\Wup|enforc|erect|raise\\Wup|reenforc)', #frontier is often meant in a methaphoric sence "frontier bulding between parties.." thus not a good word here
            '(?:^|\\W)(build|construct|set\\Wup|enforc|erect|raise\\Wup|reenforc).{0,15}\\W(barrier|border|fence|wall)', 
            '(?:^|\\W)(barrier|border|fence|frontier|wall).{0,5}\\W(control|check|inspect|monitor|protect|secur|surveill)',
            '(?:^|\\W)(control|check|inspect|monitor|protect|secur|surveill).{0,15}\\W(barrier|border|fence|frontier|wall)',
            '(?:^|\\W)immigration\\Wcontrol')




crime <- c('(?:^|\\W)crime\\W', '(?:^|\\W)criminal', '(?:^|\\W)decriminal', '(?:^|\\W)anti-crime', '(?:^|\\W)anti-criminal')


crime_types <- c('(?:^|\\W)counterfeit', 
                 '(?:^|\\W)homicid', '(?:^|\\W)manslaughter',  '(?:^|\\W)murder',
                 '(?:^|\\W)theft', '(?:^|\\W)burglar','(?:^|\\W)thieve', '(?:^|\\W)thief','(?:^|\\W)robbery', 
                 '(?:^|\\W)riot', '(?:^|\\W)brawl', '(?:^|\\W)fraud',
                 '(?:^|\\W)kidnap', '(?:^|\\W)traffick', '(?:^|\\W)smuggl', 
                 '(?:^|\\W)(child|underage|woman)\\Wabuse','(?:^|\\W)abuse\\W(child|underage)','(?:^|\\W)child\\Wporn', 
                 '(?:^|\\W)(batter|beat|hit|punch).{0,5}\\W(child|underage|woman)', '(?:^|\\W)(child|underage|woman).{0,5}\\W(batter|beat|hit|punch)',
                 '(?:^|\\W)sexual\\W(abuse|assault|harassment|offence|violation)', 
                 '(?:^|\\W)rape', '(?:^|\\W)rapist', '(?:^|\\W)harassment') #the plant "rape" is less common, "kill" is not included (was tested seperately and was mostly used for migrants being killed)


crime_types_exclude <- c('(?:^|\\W)(electoral|election|voter).{0,5}\\Wfraud')


illegal <- c('(?:^|\\W)illegal')



law <-c('(?:^|\\W)unlawful', 
        '(?:^|\\W)penal\\W(code|law)', '(?:^|\\W)strafrecht', 
        '(?:^|\\W)plead\\Wguilty',  '(?:^|\\W)prosecution', 
        '(?:^|\\W)law\\Wand\\Worder', '(?:^|\\W)zero\\Wtolerance',
        '(?:^|\\W)law.{0,5}\\W(break|enforc|violat)', '(?:^|\\W)(break|enforc|violat).{0,5}\\Wlaw\\W',
        '(?:^|\\W)judicial\\Wcontrol')


law_exclude <- c("(?:^|\\W)flee.{0,5}\\Wprosecution")



police <- c('(?:^|\\W)police', '(?:^|\\W)arrest', '(?:^|\\W)gendarmerie', 
            '(?:^|\\W)intelligence\\W(agen|servent|service)', '(?:^|\\W)secret\\Wservice',
            '(?:^|\\W)militar',  '(?:^|\\W)(civil|coast).{0,5}\\W(guard|patrol|troop)') 

police_exclude <- c("(?:^|\\W)rescue.{0,10}\\W(guard|gendarmerie|militar|police)", "(guard|gendarmerie|militar|police).{0,10}\\Wrescue",
                    '(?:^|\\W)(language|grammar)\\Wpolice', '(?:^|\\W)police\\Wstation.{0,10}\\Wask.{0,5}\\Wfor\\Wasylum', '(?:^|\\W)ask.{0,5}\\Wfor\\Wasylum.{0,10}\\Wpolice\\Wstation')  



punishment <- c('(?:^|\\W)convict', '(?:^|\\W)deport', '(?:^|\\W)detention',
                '(?:^|\\W)penalty', '(?:^|\\W)punishment','(?:^|\\W)punish', '(?:^|\\W)punitive', 
                '(?:^|\\W)jail',   '(?:^|\\W)imprison', '(?:^|\\W)prison')

punishment_exclude <- '(?:^|\\W)punish\\Wcountry'




security <- c('(?:^|\\W)insecurity', '(?:^|\\W)security',
              '(?:^|\\W)peacekeep', '(?:^|\\W)national\\W(peace|defence)','(?:^|\\W)(maintenance|maintain|keep|sustain).{0,5}\\Wpeace\\W',
              '(?:^|\\W)(bring|community|guarantee|ensure|homeland|national|nation|public).{0,5}\\Wsafe','(?:^|\\W)safe.{0,5}\\W(community|homeland|national|nation|public)')  


security_exclude <- c('(?:^|\\W)insecurity.{0,15}\\W(among|between)\\W(asylum|muslim|migrant|immigrant|refugee|foreigner|foreign\\Wpeople|foreign\\Wworker)',
                      '(?:^|\\W)security\\Wof\\W(asylum|muslim|migrant|immigrant|refugee|foreigner|foreign\\Wpeople|foreign\\Wworker|mosque)',
                      '(?:^|\\W)(look|search)\\Wfor\\Wsecurity',
                      '(?:^|\\W)(basic|financial|job|income|social)\\Wsecurity',  '(?:^|\\W)(find|look|search).{0,15}\\Wsecurity',
                      '(?:^|\\W)security.{0,15}\\Win.{0,5}\\W(afghanistan|colombia|east\\Wafrica|iraq|iran|jemen|libya|mali|mexico|middle\\Weast|nigeria|sahara|sahel|somalia|south\\Wsudan|syria|sub*saharan|sudan|pakistan|myanmar|ukraine|venezuela|yemen)',
                      '(?:^|\\W)cause\\Wof\\W(migrating|emigrating)')


terrorism <- c('(?:^|\\W)terror','(?:^|\\W)(islam|muslim|religious).{0,15}\\Wextremism')



violence <- c('(?:^|\\W)violent', 
              '(?:^|\\W)violator',
              '(?:^|\\W)violence')



weapon <- c('(?:^|\\W)firearm', '(?:^|\\W)arm\\Wcontrol',
            '(?:^|\\W)gun\\W', '(?:^|\\W)gunfire\\W','(?:^|\\W)handgun\\W','(?:^|\\W)pistol\\W',
            '(?:^|\\W)tear\\Wgas','(?:^|\\W)weapon', '(?:^|\\W)knife',
            '(?:^|\\W)stab')




other_general <- c('(?:^|\\W)felon', '(?:^|\\W)gang', '(?:^|\\W)organized\\Wband', '(?:^|\\W)delinquency', '(?:^|\\W)delinquent', '(?:^|\\W)offen(c|)se', '(?:^|\\W)offender',
                   '(?:^|\\W)juvenile\\Wdelinquen', '(?:^|\\W)abnormal\\Wlawbreaker', '(?:^|\\W)perpetrator\\Wof\\Wcrime')

#now follow two groups of words:if they appear in the same sentence with migration words and the keywords mentioned previously, a sentence is not coded as security-related anymore. 

exclude_nationalistic <- c('(?:^|\\W)(anger|crime|attack|fire|resentimment|violence|violent\\Wact).{0,10}\\W(on|against|directed\\Wat|targeted\\Wat|toward).{0,15}\\W(asylum|migrant|immigrant|refugee|foreigner|foreign\\Wpeople|foreign\\Wworker)',
                           '(?:^|\\W)xenoph', '(?:^|\\W)nazi\\W', '(?:^|\\W)neo.?nazi\\W', 
                           '(?:^|\\W)national\\Wsocialist\\Wunderground', '(?:^|\\W)nsu\\W', 
                           '(?:^|\\W)racis', '(?:^|\\W)islamophobi', '(?:^|\\W)anti.?islamist',
                           '(?:^|\\W)(right.?wing).{0,5}\\Wextremis', '(?:^|\\W)vigilante\\Wmob', '(?:^|\\W)(racial|ethnic|religious).{0,5}\\Whatred', 
                           '(?:^|\\W)hate\\Wcrime')

exclude_victim <- c('(?:^|\\W)(defend|protect|rescue).{0,5}\\W(asylum|migrant|immigrant|refugee|foreigner|foreign\\Wpeople|foreign\\Wworker|mosque)',
                    '(?:^|\\W)(asylum|migrant|immigrant|refugee|foreigner|foreign\\Wpeople|foreign\\Wworker).{0,5}\\W(defend|protect|rescue)',
                    '(?:^|\\W)slavery', '(?:^|\\W)human\\Wdrama',  '(?:^|\\W)ethnic\\Wcleansing', '(?:^|\\W)humanitarian\\Wcrisis', '(?:^|\\W)exodus\\Wof\\Wrefugee', 
                    '(?:^|\\W)(flee|escape).{0,5}\\Wfrom\\W(terror|violence|war)', '(?:^|\\W)seek.{0,5}\\Wrefuge\\Win\\Weurope', '(?:^|\\W)(flee|escape)\\Wto\\Weurope')

###########
## create dictionary
###########

dict_name <- c("migration", 
               "border", "crime", "crime_types", 'crime_types_exclude', 
               'illegal',  'law', 'law_exclude', 
               'police', 'police_exclude',
               'punishment', 'punishment_exclude',
               'security', 'security_exclude', 
               'terrorism',  'violence', 'weapon', 'other_general', 
               'exclude_nationalistic', 'exclude_victim') 



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


count_per_dict_sentence$border_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$border >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$border >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$border >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$border_combi[is.na(count_per_dict_sentence$border_combi)] <- 0 

count_per_dict_sentence$crime_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$crime >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$crime >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$crime >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$crime_combi[is.na(count_per_dict_sentence$crime_combi)] <- 0 #


count_per_dict_sentence$crime_types_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$crime_types >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$crime_types >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$crime_types_exclude >=1 & count_per_dict_sentence$crime_types >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$crime_types >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$crime_types_combi[is.na(count_per_dict_sentence$crime_types_combi)] <- 0 


count_per_dict_sentence$illegal_combi <- case_when(
  count_per_dict_sentence$illegal >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$illegal_combi[is.na(count_per_dict_sentence$illegal_combi)] <- 0 


count_per_dict_sentence$law_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$law >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$law >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$law_exclude >=1 & count_per_dict_sentence$law >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$law >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$law_combi[is.na(count_per_dict_sentence$law_combi)] <- 0 


count_per_dict_sentence$police_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$police >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$police >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$police_exclude >=1 & count_per_dict_sentence$police >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$police >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$police_combi[is.na(count_per_dict_sentence$police_combi)] <- 0 


count_per_dict_sentence$punishment_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$punishment >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$punishment >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$punishment_exclude >=1 & count_per_dict_sentence$punishment >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$punishment >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$punishment_combi[is.na(count_per_dict_sentence$punishment_combi)] <- 0 


count_per_dict_sentence$security_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$security >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$security >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$security_exclude >=1 & count_per_dict_sentence$security >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$security >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$security_combi[is.na(count_per_dict_sentence$security_combi)] <- 0 


count_per_dict_sentence$terrorism_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$terrorism >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$terrorism >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$terrorism >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$terrorism_combi[is.na(count_per_dict_sentence$terrorism_combi)] <- 0 

count_per_dict_sentence$violence_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$violence >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$violence >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$violence >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$violence_combi[is.na(count_per_dict_sentence$violence_combi)] <- 0 

count_per_dict_sentence$weapon_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$weapon >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$weapon >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$weapon >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$weapon_combi[is.na(count_per_dict_sentence$weapon_combi)] <- 0 

count_per_dict_sentence$other_general_combi <- case_when(
  count_per_dict_sentence$exclude_nationalistic >=1 & count_per_dict_sentence$other_general >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$exclude_victim >=1 & count_per_dict_sentence$other_general >=1 & count_per_dict_sentence$migration >=1 ~ 0,
  count_per_dict_sentence$other_general >=1 & count_per_dict_sentence$migration >=1 ~ 1)
count_per_dict_sentence$other_general_combi[is.na(count_per_dict_sentence$other_general_combi)] <- 0 



#check results on sentence level
# add sentence id for merging (order is correct)


count_per_dict_sentence$reminderid_sent_id <- rownames(count_per_dict_sentence)
corpus_sentence$reminderid_sent_id <- rownames(corpus_sentence)

#merge sentence and hits
corpus_sentence_with_dict_hits <- merge(corpus_sentence, count_per_dict_sentence, by = "reminderid_sent_id")


#calc fr_sec variable
corpus_sentence_with_dict_hits$fr_sec <- case_when(corpus_sentence_with_dict_hits$border_combi >= 1 |corpus_sentence_with_dict_hits$crime_combi >= 1 | corpus_sentence_with_dict_hits$crime_types_combi >= 1| corpus_sentence_with_dict_hits$illegal_combi >= 1| corpus_sentence_with_dict_hits$law_combi >= 1|
                                                              corpus_sentence_with_dict_hits$police_combi >= 1| corpus_sentence_with_dict_hits$punishment_combi >= 1|  corpus_sentence_with_dict_hits$security_combi  >= 1| corpus_sentence_with_dict_hits$terrorism_combi >= 1| corpus_sentence_with_dict_hits$weapon_combi >= 1 |corpus_sentence_with_dict_hits$other_general_combi >= 1~ 1)#




#aggregate the sentence level hit results on article level 

corpus_with_fr_agg <- corpus_sentence_with_dict_hits %>% 
  group_by(sample_ID) %>% 
  summarise_at(vars("fr_sec"), mean)


corpus_new <- merge(corpus_with_fr_agg, corpus, by = "sample_ID", all =T)


corpus_new <- subset(corpus_new, select = c(sample_ID, fr_sec))
corpus_new <- corpus_new %>% mutate(fr_sec = if_else(is.na(fr_sec), 0, fr_sec)) #set NA to 0 (necessary for sum within group later)

table(corpus_new$fr_sec) # descriptive overview


######
#save annotated dataset
###########
write.csv(corpus_new, "fr_sec.csv") # 




