##PART 0: PRELIMINARY HOUSEKEEPING
#Set up working directory
setwd("C:/Schriftliches/STUDIUM Bayreuth/Semester III/Data Mining mit R/Uebungsprojekte/Titanic")

#Load libraries
library(tidyverse)
library(caret)
library(GGally)
library(magrittr)

##PART 1: DATA INSPECTION
#Load in data
train <- read.csv("C:/Schriftliches/STUDIUM Bayreuth/Semester III/Data Mining mit R/Uebungsprojekte/Titanic/train.csv", 
                  row.names=1)
test <- read.csv("C:/Schriftliches/STUDIUM Bayreuth/Semester III/Data Mining mit R/Uebungsprojekte/Titanic/test.csv", 
                 row.names=1)

#Inspect column types
glimpse(train)
#Convert column types
train_clean <- train %>%
  as_tibble()%>%
  mutate(across(.cols = c(Name, Ticket, Cabin), .fns = as.character))%>%
  mutate(across(.cols = c(Survived, Pclass), .fns = as_factor))
#inspect column properties
summary(train_clean)
#Many NAs in Age
#Slight imbalance of Y variable in favour of 'not survived'
#Many outliers in Fare
#Strong left-skew in Parch and SibSp

#Convert column names to lower for convenience
train_clean %<>%
  rename_with(.fn = tolower)


#first visual summary
train_clean %>%
  select(-c(name, ticket, cabin))%>%
  ggpairs()

#Count NAs in numeric columns that have any
train_clean %>%
  select(where(~ any(is.na(.x))))%>%
  summarise(across(.fns = ~ sum(is.na(.x))))%>%
  #Add empty strings count of character variables
  cbind(
    train_clean %>%
      summarise(across(.cols = where(is_character), .fns = ~ sum(.x == "")))
  )
#Only Age has NAs, but it is a considerable amount.
#We might have to interpolate age
#Cabin has a majority of empty strings


##PART 2: Data cleaning and preparation
#Let's look at columns one at a time, since there are not too many

#Explore Y-variable: survived
sum(train_clean$survived==1)/nrow(train_clean)
#The dependent variable is unbalanced, there are about 39% survivors and 61% victims

#Explore Column 1: pclass
ggplot(train_clean, aes(x=pclass, fill=survived))+
  geom_bar()
#Passengers in third class seem to perish more frequently and survive better in first class
#This seems like a promising variable

#Explore Column 2: name
#From domain knowledge we can see that passengers' names contain important information
#People have titles (Mr. or Miss. that indicate e.g. gender)
#We can return to this column later to extract features

#Explore Column 3: sex
ggplot(train_clean, aes(x=sex, fill=survived))+
  geom_bar()
#Males seem to perish considerably more than females

ggplot(train_clean, aes(x=sex, fill=survived))+
  geom_bar()+
  facet_wrap(~pclass)
#This trend seems to be consistent across pclasses
#Quick check: Survival rates
train_clean %>%
  group_by(pclass, sex)%>%
  summarise(survival_rate = sum(as.numeric(survived))/n())
#Confirmed: Survival rate decreases in pclass and in male-ness

#Explore column 4: age
ggplot(train_clean, aes(x=cut_interval(age, n=20), fill=survived))+
  geom_bar(position="fill")
#Seems like very young and very old passengers are more likely to survive with no clear cut-off point

ggplot(train_clean, aes(x=age, fill=survived))+
  geom_histogram()+
  facet_grid(sex~pclass)
#This seems to roughly hold across pclass and sex, but age seems to be a much weaker predictor

as.tibble(train_clean %>%
            mutate(age = cut_interval(age, n=5))%>%
            group_by(sex, pclass, age)%>%
            summarise(survival_rate = sum(as.numeric(survived))/n()))%>%
  ggplot(aes(x=age, y=survival_rate))+
  geom_col()+
  facet_wrap(sex~pclass)
#Survival rates for females in class 1 and 2 are high regardless of age
#Survival rates for makes in class 1 and 2 seem to be higher for younger males
#BUT: Age has many NAs so we need to figure out a way to impute them for later, as it seems to be a promising variable

#Explore column 5: sibsp
ggplot(train_clean, aes(x=sibsp, fill=survived))+
  geom_bar(position="fill")
#Apparenty, chances of survival are generally decreasing with an increase in sibsp
#Peak survival at 1 sibsp

ggplot(train_clean, aes(x=sibsp, fill=survived))+
  geom_bar()+
  facet_wrap(sex~pclass)
#Many males in class 3 had very few siblings or spouses on board

ggplot(train_clean, aes(x=sibsp, fill=survived))+
  geom_bar(position="fill")+
  facet_wrap(sex~pclass)
#Sibsp seems to be a weaker indicator, but seems to eplain the deaths of females in third class: the more sibsp the worse

#Explore column 6: parch
ggplot(train_clean, aes(x=parch, fill=survived))+
  geom_bar(position="stack")
#Too many children seem to be bad overall
#Majority travelled without children, and very left-skewed distribution with almost no records for parch >2


#Explore column 7: ticket
#Ticket is a character string
length(unique(train_clean$ticket)) == nrow(train_clean)
#Some passengers were travelling under the same ticket number, e.g. probably children and parents
#Ticket is a messy variable containing numbers and letters with no apparent system at first glance
#We'll have to dig deeper into Ticket in the feature engineering section

#Explore column 8: fare
train_clean %>%
  ggplot(aes(x=survived, y=fare, fill=survived))+
  geom_violin()+
  scale_y_log10()+
  geom_boxplot(width = 0.1, color="grey", alpha = 0.5)
#Those who survived tend to have paid a higher fare and those who paid the cheapest fares did not survive
#BUT: A fare of 0 could also mean "ticket was gifted" or "is an employee of the company" or so. Since there are only very few with fare 0, we can neglect them for the moment

#One thing to stay aware of is that Fare and PClass might be describing the same effect:
#Who pays more tends to be in a higher class
ggplot(train_clean, aes(x=pclass, y=fare))+
  geom_boxplot(width=0.2, alpha = 0.5)+
  geom_violin(alpha=0.5)+
  coord_cartesian(ylim = c(0,70))
cor(as.integer(train_clean$pclass), train_clean$fare, method = "spearman")

#Explore column 9: cabin
table(train_clean$cabin)
#Messy variable: sometimes multiple cabins are indicated in a single row
#There seems to be a system to them: [A-G]+[Integer]
#We have to dig deeper into this in the feature engineering section

#Explore column 10: embarked
table(train_clean$embarked)
ggplot(train_clean, aes(x=embarked, fill=survived))+
  geom_bar()
#Most people embarked in S, C and Q are little in proportion
total_num_survived <- sum(train_clean$survived==1)

train_clean %>%
  group_by(embarked)%>%
  summarise(pop_share = n()/nrow(train_clean),
            surv_share = sum(survived==1)/total_num_survived,
            diff = abs(pop_share - surv_share))
#Each embarkment category's share of survivors corresponds to its total share of all passengers
#Seems to be uninteresting at first glance as a direct x-variable

#PART 3: PRE-MODELLING
#Feature selection candidates


#Feature engineering
#Impute age from title

#Disentangle Ticket IDs

#Create CV folds



#PART 4: MODELLING