# Named Entity Recognition Tagging on CLINICAL DATA

## Topic of Contents
* Overview
* Motivation
* Demo
* Technical Aspects
* Setup

## Overview

The project focuses on building pipelines for data-preprocessing, training and testing using airflow. The goal of the project is to identify the diseases and treatments 
from the input text. Ex:

Clinical Note: “The patient was a 62-year-old man with squamous cell lung cancer, which was first successfully treated by a combination of radiation therapy and chemotherapy.”
Disease:  Lung Cancer
Treatment: Radiation Therapy and Chemotherapy

A conditional random field model is used to train on the text data to predict the NER Tags: O,T, and (i.e others, treatments and diseases) respectively.

## Motivation

The project is a case study on a health services based web platform , which enables doctors to list their services and manage patient interactions. It provides various services for patients, such as booking interactions with doctors and ordering medicines online.  (Similar to 1mg, PharmEasy). Assuming that the app is widely used by millions of people, there is a lot of clinical text present in e-presciptions. We can perform text analytics on this free clinical text and gain insights. 

Currently, if you need to extract diseases and treatments from free text, a trained person with clinical knowledge must manually look at the clinical notes and then pull this information. 
A data entry team would manually look at the clinical text and extract information about diseases and their treatment data. A data validation team would validate the extracted information. This process is prone to errors, and as the data increases, the data-entry team’s size would need to be scaled up.
Automating this process would result in reduced man-hours. The data-entry team would not be required. The data validation team would still need to perform validation on extracted data. It would also significantly reduce manual errors.

Hence the business KPI addressed in this project is to automate the extraction of diseases and treatments thus reducing man-hours involved in the task.

## Technical Aspect

* Data pre-processing 
 - the train sentence had each word in one line. The first step was to form sentences. The pre-processing was applied on the labels data.
 - EDA on most of common nouns and pronouns used
 ('patients', 492), ('treatment', 281), ('%', 247), ('cancer', 200), 
('therapy', 175), ('study', 154), ('disease', 142), ('cell', 140), 
('lung', 116), ('group', 93)

 - Following features were extracted to feed into to the CRF model :
 
 word.lower','word.isupper','word.pos','word[-3:]', 'word[-2:]',
 'word.digit','word.startsWithCapital' ,'word.previous_pos',
 'word.previous.startsWithCapital','word.previous.digit',
'word.previous.isupper','word.beg’,'word.end
 
 * CRF model from sklearn_crfsuite module was used to train on the features. Model performed well with a F1 score of 0.89.
 
 * The pre-processing, training and testing steps were automated using airflow DAGS. The model was tracked in mlflow.
 
## Setup 


