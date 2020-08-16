import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re,nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve



def cleaned_reviews(review):
    """
    This function will remove the unncessary 
    symbols from a review such as punctuation mark, numbers ,emoji. 
    
    Args:
        review: str
        
    Returns:
        cleaned review: str
    """
    review = review.replace('\n', '') #removing new line 
    review = re.sub('[^\u0980-\u09FF]',' ',str(review)) #removing unnecessary punctuation
    return review
                                                 #=========================== 
                                                 ####### Function 2 #########
                                                 #===========================  
def stopwords_info(filename):
    """
    This function will create a stopwords list from the (.txt) file. 
    
    Args:
        filename: str
        
    Returns:
        stp: list of stopwords
        num_of_words: int
    """
    stp = open(filename,'r',encoding='utf-8').read().split()
    num_of_stopwords = len(stp)
    return stp,num_of_stopwords

                                                 #=========================== 
                                                 ####### Function 3 #########
                                                 #===========================  
def stopword_removal(review,stopwords):
    """
    This function will remove the stopwords from a review. 
    
    Args:
        review: str
        stopwords: str
        
    Returns:
        review without stopwords: str
    """
    stp,num_of_stopwords =stopwords_info(stopwords)
    result = review.split()
    reviews = [word.strip() for word in result if word not in stp ]
    reviews =" ".join(reviews)
    return reviews
                                                 #=========================== 
                                                 ####### Function 4 #########
                                                 #===========================               
def process_reviews(review,stopwords,removing_stopwords):
    """
    This function will process all the reviews which includes
    cleaning text and stopword removal.
    
    Args:
        review: str
        stopwords: filename
        removing_stopwords: boolean(True or False) 
        
    Returns:
        cleaned reviews: str
    """
    if removing_stopwords ==False:
        reviews = cleaned_reviews(review)
        
    else:
        reviews = cleaned_reviews(review)
        reviews = stopword_removal(reviews,stopwords)
        
    return reviews    



                                                 #=========================== 
                                                 ####### Function 5 #########
                                                 #===========================  

def data_summary(dataset):
    
    """
    This function will print the summary of the reviews and words distribution in the dataset. 
    
    Args:
        dataset: list of cleaned sentences   
        
    Returns:
        Number of documnets per class: int 
        Number of words per class: int
        Number of unique words per class: int
    """
    documents = []
    words = []
    u_words = []
    class_label= [k for k,v in dataset.Sentiment.value_counts().to_dict().items()]
  # find word list
    for label in class_label: 
        word_list = [word.strip().lower() for t in list(dataset[dataset.Sentiment==label].cleaned) for word in t.strip().split()]
        counts = dict()
        for word in word_list:
                counts[word] = counts.get(word, 0)+1
    # sort the dictionary of word list  
        ordered = sorted(counts.items(), key= lambda item: item[1],reverse = True)
    # Documents per class
        documents.append(len(list(dataset[dataset.Sentiment==label].cleaned)))
    # Total Word per class
        words.append(len(word_list))
    # Unique words per class 
        u_words.append(len(np.unique(word_list)))
       
        print("\nClass Name : ",label)
        print("Number of Documents:{}".format(len(list(dataset[dataset.Sentiment==label].cleaned))))  
        print("Number of Words:{}".format(len(word_list))) 
        print("Number of Unique Words:{}".format(len(np.unique(word_list)))) 
        print("Most Frequent Words:\n")
        for k,v in ordered[:10]:
              print("{}\t{}".format(k,v))
   
    return documents,words,u_words,class_label


                                       #==================================================
                                       ################# Label Encoding Function #########
                                       #==================================================
def label_encoding(sentiment,bool):
    """
    This function will return the encoded labels in array format. 
    
    Args:
        sentiment: series of class names(str)
        bool: boolean (True or False)
        
    Returns:
        labels: numpy array 
    """
    le = LabelEncoder()
    le.fit(sentiment)
    encoded_labels = le.transform(sentiment)
    labels = np.array(encoded_labels) # Converting into numpy array
    class_names =le.classes_ ## Define the class names again
    if bool == True:
        print("\n\t\t\t===== Label Encoding =====","\nClass Names:-->",le.classes_)
        for i in sample_data:
            print(sentiment[i],' ', encoded_labels[i],'\n')

    return labels  


                           #===========================================================
                           ################# Dataset Splitting Function ###############
                           #=========================================================== 

def dataset_split(feature_space,sentiment):
    """
    This function will return the splitted (80%-20%) feature vector . 
    
    Args:
        feature_space: calcuated feature vector (sparse matrix)
        sentiment: encoded lables (array) 
        
    Returns:
        X_train: training feature vector (sparse matrix)
        X_test : testing feature vector (sparse matrix)
        y_train: training encoded labels (array) 
        y_test : testing encoded labels (array) 
    """

    X_train,X_test,y_train,y_test = train_test_split(feature_space,sentiment,train_size = 0.8,
                                                  test_size = 0.2,random_state =0)
    print("Feature Size :======>",X_train.shape[1])
    print("\nDataset Distribution:\n")
    print("\tSet Name","\t\tSize")
    print("\t========\t\t======")

    print("\tFull\t\t\t",feature_space.shape[0],
        "\n\tTraining\t\t",X_train.shape[0],
        "\n\tTest\t\t\t",X_test.shape[0])
  
    return X_train,X_test,y_train,y_test

                                            #======================================
                                            ##### Unigram Tf-idf value calculation
                                            #======================================
            
def calc_unigram_tfidf(reviews):
    """
    This function will return the tf-idf value of the unigram features . 
    
    Args:
        reviews: a list of cleaned reviews   
        
    Returns:
        tfidf: a instance of TfidfVectorizer
        X : Unigram Feature Vector (sparse matrix)
    """
    tfidf = TfidfVectorizer(use_idf=True,tokenizer=lambda x: x.split()) 
    X = tfidf.fit_transform(reviews)
    
    return tfidf,X

                                            #======================================
                                            ##### Bi-gram Tf-idf value calculation
                                            #======================================
            
def calc_bigram_tfidf(reviews):
    """
    This function will return the tf-idf value of the bigram features . 
    
    Args:
        reviews: a list of cleaned reviews   
        
    Returns:
        tfidf: a instance of TfidfVectorizer
        X : Bigram Feature Vector (sparse matrix)
    """
    tfidf = TfidfVectorizer(ngram_range=(1,2),use_idf=True,tokenizer=lambda x: x.split()) 
    X = tfidf.fit_transform(reviews)
    
    return tfidf,X

                                            #======================================
                                            ##### Tri-gram Tf-idf value calculation
                                            #======================================


def calc_trigram_tfidf(reviews):
    """
    This function will return the tf-idf value of the bigram features . 
    
    Args:
        reviews: a list of cleaned reviews   
        
    Returns:
        tfidf: a instance of TfidfVectorizer
        X : Tri-gram Feature Vector (sparse matrix)
    """
    tfidf = TfidfVectorizer(ngram_range=(1,3),use_idf=True,tokenizer=lambda x: x.split()) 
    X = tfidf.fit_transform(reviews)
    
    return tfidf,X

                                    #=======================
                                    ##### Print Tf-idf value
                                    #=======================

def show_tfidf(tfidf_instance,samp_review): 
    """
    This function will print the tfidf value of a specific review . 
    
    Args:
        tfidf_instance: an object
        samp_review   : string
        
    Returns:
        tfidf value of gram feature 
    """
    print("Sample Review: ",samp_review)
    first_vector = tfidf_instance.transform([samp_review])
    df = pd.DataFrame(first_vector.T.todense(), index=tfidf_instance.get_feature_names(), columns=["tfidf"])
    a = df.sort_values(by=["tfidf"],ascending=False)
    print(a[0:len(samp_review.split())])
    
    
                                              #============================================
                                              #########  Classifiers for Unigram  #########
                                              #============================================
def ml_models_for_unigram_tfidf(): 
    """
    This function consists the models defination for Unigram Features
    
    Retuns:
        ml_models: list of models
        model_names: list of model_names
    
    """
    
    lr_model = LogisticRegression(random_state = 123)
    dt_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    rf_model = RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state = 0)
    mnb_model = MultinomialNB(alpha=0.15)
    knn_model = KNeighborsClassifier(n_neighbors=3, metric = 'minkowski')
    lsvm_model = SVC(kernel = 'linear',C = 0.2, probability=True, random_state = 0)
    ksvm_model = SVC(C= 1000,kernel = 'rbf',probability=True, gamma = 0.00015, random_state = 0)
    sgd_model = SGDClassifier(loss ='log',penalty='l2', max_iter=5)
    model_names = ['Logistic Regression','Decision Tree','Random Forest','Naive Bayes','KNN','Linear SVM','Kernel SVM','SGD']
    # Create list of models
    ml_models = [lr_model,dt_model,rf_model,mnb_model,knn_model,lsvm_model,ksvm_model,sgd_model]
    return ml_models,model_names    

    
                                              #============================================
                                              #########  Classifiers for Bigram  #########
                                              #============================================
def ml_models_for_bigram_tfidf(): 
    """
    This function consists the models defination for Bigram Features
    
    Retuns:
        ml_models: list of models
        model_names: list of model_names
    
    """
    lr_model = LogisticRegression(random_state = 123)
    dt_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    rf_model = RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state = 0)
    mnb_model = MultinomialNB(alpha=0.008)
    knn_model = KNeighborsClassifier(n_neighbors=2, metric = 'minkowski')
    lsvm_model = SVC(kernel = 'linear',C = 0.3, probability=True, random_state = 0)
    ksvm_model = SVC(C= 1000,kernel = 'rbf',probability=True, gamma = 0.00015, random_state = 0)
    sgd_model = SGDClassifier(loss = 'log',penalty='l2', max_iter=5)
    model_names = ['Logistic Regression','Decision Tree','Random Forest','Naive Bayes','KNN','Linear SVM','Kernel SVM','SGD']
    ml_models = [lr_model,dt_model,rf_model,mnb_model,knn_model,lsvm_model,ksvm_model,sgd_model]
    return ml_models,model_names    


                                              #============================================
                                              #########  Classifiers for Trigram  #########
                                              #============================================
def ml_models_for_trigram_tfidf(): 
    """
    This function consists the models defination for Tri-gram Features
    
    Retuns:
        ml_models: list of models
        model_names: list of model_names
    
    """
    
    lr_model = LogisticRegression(random_state = 123)
    dt_model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    rf_model = RandomForestClassifier(n_estimators=100, criterion ='entropy', random_state = 0)
    mnb_model = MultinomialNB(alpha=0.001)
    knn_model = KNeighborsClassifier(n_neighbors=2, metric = 'minkowski')
    lsvm_model = SVC(kernel = 'linear',C = 0.3, probability=True, random_state = 0)
    ksvm_model = SVC(C= 1000,kernel = 'rbf',probability=True, gamma = 0.0002, random_state = 0)
    sgd_model = SGDClassifier(loss = 'log',penalty='l2', max_iter=5)
    model_names = ['Logistic Regression','Decision Tree','Random Forest','Naive Bayes','KNN','Linear SVM','Kernel SVM','SGD']
    #create a list of models
    ml_models = [lr_model,dt_model,rf_model,mnb_model,knn_model,lsvm_model,ksvm_model,sgd_model]
    return ml_models,model_names     
    
    
                                           #============================================
                                           ####### Model Evaluation Function ############
                                           #=============================================

def model_performace(model,X_train,X_test,y_train,y_test):
    """
    This function will return the performance parameter values of each ML models.
    Performance parameters are Accuracy, F1-Score, Precision, Recall.
    
    Args:
        model: a ML model instance
        X_train: training feature vector (sparse matrix)
        X_test : testing feature vector (sparse matrix)
        y_train: training encoded labels (array) 
        y_test : testing encoded labels (array) 
        
    Returns:
        my_dict: a dictionary of all the parameters for each models
    """
    my_dict = {}
    model.fit(X_train,y_train)
    # Prediction
    pred_y = model.predict(X_test)
    my_dict['Accuracy'] = round(accuracy_score(y_test, pred_y),4)*100 
    my_dict['Precision'] = round(precision_score(y_test, pred_y),4)*100 
    my_dict['Recall'] = round(recall_score(y_test, pred_y),4)*100 
    my_dict['F1 Score'] = round(f1_score(y_test, pred_y),4)*100 
    
    return my_dict  

                                        #========================================
                                        #### Model Performane into Dataframe #####
                                        #=========================================
            
def performance_table(performance_dict):
    """
    This function will create a dataframe of all the performance parameters.
    
    Args:
        performance_dict: a dictionary of all the parameters for each models
        
    Returns:
        performance_df: a dataframe
    """

    acc_list = []
    pr_list = []
    re_list = []
    f1_list = []
    for i in performance_dict.keys():
        acc_list.append(performance_dict[i]['Accuracy'])
        pr_list.append(performance_dict[i]['Precision'])
        re_list.append(performance_dict[i]['Recall'])
        f1_list.append(performance_dict[i]['F1 Score'])

    # Create a dataframe
    performance_df = pd.DataFrame({'Accuracy':acc_list,'Precision':pr_list,
                                   'Recall':re_list,'F1 Score':f1_list},
                                  index =['LR','DT','RF','MNB','KNN','Linear SVM','RBF SVM','SGD'])
    return performance_df

                                        #============================
                                        #### Plotting ROC Curve #####
                                        #============================



def plot_roc_curve(gram_models,X_train,X_test,y_train,y_test,gram_name):
    
    """
    This function will plot the ROC curve for all classifiers
    
    Args:
        gram_models: a function of all the models defination for a gram feature
        X_train: training feature vector (sparse matrix)
        X_test : testing feature vector (sparse matrix)
        y_train: training encoded labels (array) 
        y_test : testing encoded labels (array) 
        gram_name: gram feature name(str)
        
    
    """
    
    ml_models,model_names = gram_models 

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    # Train the models and record the results
    for i,model in enumerate(ml_models):
        model = model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::,1]

        fpr, tpr, _ = roc_curve(y_test, y_pred)

        auc = roc_auc_score(y_test, y_pred)

        result_table = result_table.append({'classifiers':model_names[i],
                                            'fpr':fpr, 
                                            'tpr':tpr, 
                                            'auc':auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    # plotting
    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                 result_table.loc[i]['tpr'], 
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=12)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=12)

    plt.title(f'ROC Curve Analysis for {gram_name} features', fontweight='bold', fontsize=12)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
    
    
                                        #=========================================
                                        #### Plotting Precision-Recall Curve #####
                                        #========================================    
    
    
    
def plot_PR_curve(gram_models,X_train,X_test,y_train,y_test,gram_name):
    """
    This function will plot the Precision Recall curve for all classifiers
    
    Args:
        gram_models: a function of all the models defination for a gram feature
        X_train: training feature vector (sparse matrix)
        X_test : testing feature vector (sparse matrix)
        y_train: training encoded labels (array) 
        y_test : testing encoded labels (array) 
        gram_name: gram feature name(str)
        
    
    """
    
    ml_models,model_names = gram_models 
    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'precision','recall','AP'])

    # Train the models and record the results
    for i,model in enumerate(ml_models):
        model = model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[::,1]

        precision, recall,_ = precision_recall_curve(y_test, y_pred)

        average_precision = average_precision_score(y_test, y_pred)

        result_table = result_table.append({'classifiers':model_names[i],
                                            'precision':precision, 
                                            'recall':recall, 
                                            'AP': average_precision}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    # Plotting

    fig = plt.figure(figsize=(8,6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['recall'], 
                 result_table.loc[i]['precision'], 
                 label="{}, AP={:.3f}".format(i, result_table.loc[i]['AP']))

    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Recall", fontsize=12)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("Precision", fontsize=12)

    plt.title(f'PR Curve Analysis for {gram_name} features', fontweight='bold', fontsize=12)
    plt.legend(prop={'size':13}, loc='lower right')

    plt.show()
    


    
    
