#extracting the training model to a file
import joblib

joblib.dump(model,'file_name')      #extract model to 'file_name'

calling_model= joblib.load('file_name') #load model from 'file_name'
