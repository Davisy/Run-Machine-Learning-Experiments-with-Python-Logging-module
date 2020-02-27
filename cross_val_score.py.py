# import important packages
from imports import *
from logs import log
from preprocessing import *

# set a logger file
logger = log(path="logs/", file="cross_val.logs")

#load dataset 
data = pd.read_csv("data/loans_data.csv")

# preprocessing the loan data 
data = preprocessing(data)


# split data into train and test
X = data.drop('Loan_Status',axis = 1)
y = data.Loan_Status

# create a dictionary for  classifiers 
models = {
    "KNN": KNeighborsClassifier(),
    "RF": RandomForestClassifier(),
    "GB": GradientBoostingClassifier(),
    "DTC": DecisionTreeClassifier(),
    "BC": BaggingClassifier(),
    "XGB": XGBClassifier(),
    "EXT": ExtraTreesClassifier(),
    "LG": LogisticRegression(),
    "BBC": BalancedBaggingClassifier(),
    "EEC": EasyEnsembleClassifier(),
}

logger.info("Start Cross Validation")

for model_name, model in models.items():
  logger.info("Train {}".format(model_name))
  
  # cross_val_score for each classifier
  scores = cross_val_score(model, X, y, cv=10, scoring = 'accuracy')
  
  logger.info("The mean score for {}: {:.3f}".format(model_name, scores.mean()))
  
  logger.info("-------------------------------")
      

logger.info("Cross Validation Ends")   