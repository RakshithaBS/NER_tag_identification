import os
import spacy
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import mlflow
from sklearn.model_selection import train_test_split
import json
import logging

logging.basicConfig(level=logging.INFO)

def read_config(config_file_path):
    with open(config_file_path,'r') as file:
        return json.load(file)
    

def process_file(filename,dump_file_path,config_file_path):
    config_json= read_config(config_file_path)
    if(config_json["re_process_raw_data"]):    
        con = open(filename,"r",encoding="utf-8")
        text = con.read()
        con.close()
        sentences=text.split("\n\n")
        sentences =[sentence.replace("\n"," ") for sentence in sentences]
        joblib.dump(sentences,dump_file_path,compress=1)
    else:
        logging.info("processing flag set as false. Skipping the task")

def word2features(sentence_list,index,pos_tags):
    word = sentence_list[index]
    features ={
      'word.lower': word.lower(),
      'word.isupper':word.isupper(),
      'word.pos':pos_tags[index],
      'word[-3:]': word[-3:],
      'word[-2:]':word[-2:],
      'word.digit':word.isdigit(),
      'word.startsWithCapital': word[0].isupper()
  }

    if index > 0:
        prev_word = sentence_list[index-1]
        features.update({
        'word.previous_pos':pos_tags[index-1],
        'word.previous.startsWithCapital':prev_word[0].isupper(),
        'word.previous.digit':prev_word.isdigit(),
        'word.previous.isupper':prev_word.isupper()
         })
    else:
        if index ==0:
            features.update({'word.beg':True})
        else:
            features.update({'word.end':True})
  
    return features


def getFeatureForSentence(sentence,model):
    pos_tags=[]
    tokens = model(sentence)
    pos_tags=[token.pos_ for token in tokens]
    sentence_list = sentence.split()
    return [ word2features(sentence_list,i,pos_tags) for i in range(len(sentence_list))]


def getLabelsListForSentence(labels):
    return labels.split()


def getPreProcessedOutput(X_file_path,Y_file_path,dump_file_path,config_file_path,training=False):
    config_json= read_config(config_file_path)
    if(config_json["re_process_features"]):
        sentences = joblib.load(X_file_path)
        labels = joblib.load(Y_file_path)
        model = spacy.load("en_core_web_sm")
        logging.info("Started preprocessing sentences------------")
        X = [getFeatureForSentence(sentence,model) for sentence in sentences]
        logging.info("Started processing labels------------------")
        Y = [getLabelsListForSentence(label) for label in labels]
        if training:
            joblib.dump(X,dump_file_path+"/X_train", compress=1)
            joblib.dump(Y,dump_file_path+"/Y_train", compress=1)
        else:
            joblib.dump(X,dump_file_path+"/X_test", compress=1)
            joblib.dump(Y,dump_file_path+"/Y_test", compress=1)
    else:
        print("processing flag set as false. Skipping the task")
    
def trainModel(tracking_uri,experiment_name,training_file_path,config_file_path):
    config_json= read_config(config_file_path)
    if(config_json['train_model']):
        X_train=joblib.load(training_file_path+'/X_train')
        Y_train=joblib.load(training_file_path+'/Y_train')
        mlflow.set_tracking_uri(tracking_uri)
        try:
            logging.info("creating the experiment")
            mlflow.create_experiment(experiment_name)
        except:
            pass
        logging.info("Setting the experiment")
        mlflow.set_experiment(experiment_name)
        X_train,X_test,Y_train,Y_test= train_test_split(X_train,Y_train,random_state=42,test_size=0.1)
        with mlflow.start_run(run_name=experiment_name) as run:
            crf = sklearn_crfsuite.CRF(max_iterations=100)
            crf.fit(X_train,Y_train) 
            y_pred = crf.predict(X_test)
            f1_score=metrics.flat_f1_score(Y_test, y_pred, average='weighted')
            joblib.dump(f1_score,training_file_path+"/validated_f1_score")
            joblib.dump(crf,training_file_path+"/trained_model")
            mlflow.log_metric('flat_f1_score',f1_score)
            mlflow.sklearn.log_model(artifact_path="models",sk_model=crf)
    
    
    
def evaluateModel(model_path,test_file_path):
    X_test=joblib.load(test_file_path+'/X_test')
    Y_test=joblib.load(test_file_path+'/Y_test')
    model=mlflow.sklearn.load_model(model_path)
    y_pred = model.predict(X_test)
    f1_score=metrics.flat_f1_score(Y_test, y_pred, average='weighted')
    joblib.dump(f1_score,test_file_path+"/test_f1_score")
    if(f1_score>0.8):
        logging.info(f"f1 score for test data: {f1_score}")
    else:
        logging.info("Model performance is below expected value.Please Re-trigger Training pipeline")
    
def checkIfDeployModel(file_path):
    test_f1_score = joblib.load(file_path+"/test_f1_score")
    if test_f1_score>0.8:
        return "deploy_model"
    else:
        logging.info("model is not upto the mark")
        return "task_failure"

def get_dictionary(y_pred,test_sentences):

    diseases_and_treatments =  {} # dictionary with disease as key an list of treatments as value
    for i in range(len(y_pred)): # For each predicted sequence
        labels = y_pred[i]
        disease = "";
        treatment = "";
        for j in range(len(labels)): # for each individual label in the sequence
            if labels[j] == 'O': # ignore if label is O -- other
                continue

            if(labels[j] == 'D'): # Label D indicates disease, so add the corresponding word from test sentence to the disease name string
                disease += test_sentences[i].split()[j] + " "
                continue

            if(labels[j] == 'T'): # Label T indicates disease, so add the corresponding word from test sentence to the treatment name string
                treatment += test_sentences[i].split()[j] + " "
                #print(treatment)
    

            #disease = disease.strip() # to remove extraneous spaces
            #treatment = treatment.strip()
            # add the identified disease and treatment to the dictionary
            # if it is a new disease, directly add the value
            # if the disease has been seen previously, get the treatment list
            # and add current treatment to the list.
            if disease is not "" and treatment is not "":
                if disease not in diseases_and_treatments.keys():
                    diseases_and_treatments[disease] = [treatment]
                else:
                    treatment_list = diseases_and_treatments.get(disease)
                    if(treatment_list!=treatment):
                        treatment_list.append(treatment)
                    diseases_and_treatments[disease] = treatment_list
    
    
    return diseases_and_treatments
    
    
def getPredition(sentence,model_path):
    # predict the NER tag for a input sentence
    X = [getFeatureForSentence(sentence) for sentence in sentences]
    model=mlflow.load_model(model_path)
    Y_pred = model.predict(X)
    disease_and_treatment=get_dictionary(Y_pred,sentence)
    logging.info(f"Disease: {disease_and_treatment.keys()}" +"\n")
    logging.info(f"Treatment: {disease_and_treatment.values()}")
    return disease_and_treatment