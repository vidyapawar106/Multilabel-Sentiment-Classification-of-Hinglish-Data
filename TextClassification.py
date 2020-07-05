#importing the pandas library to load and process the dataset
import pandas as pd
names= ['id','text','label'] #Naming the Columns
train_df = pd.read_csv(r'C:\Users\vidyapawar106\Downloads\Nishafin.csv',names=names, sep=",")
eval_df = pd.read_csv(r'C:\Users\vidyapawar106\Downloads\kabitakitchen.csv',names=names, sep=",")
data = train_df.append(eval_df) #Combining two datasets obtained from two different Youtube Channels
del data['id'] #This column is not required therefore eliminating it completely
#Replacing all the Punctuations and unnecessary Ascii Charcters
data['text'] = data['text'].apply(lambda x: x.replace('\\', ' '))
data['text'] = data['text'].str.replace('[^\w\s#@/:%.,_-]', ' ')
data['text'] = data['text'].str.replace('[^A-Za-z0-9  ]+', ' ')
data['label'] = data['label'].apply(lambda x:x-1)

#importing regex library for string operations
import re 
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # removing emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
data['text'] = [emoji_pattern.sub(r' ', e) for e in data['text']]

#Splitting dataset into train and eval with 75:25 ratio
from sklearn.model_selection import train_test_split
train, test, y_train, y_test = train_test_split(data, data['label'], test_size=0.25, stratify=data['label'],
                                                    random_state=123, shuffle=True)

#Loading the Simple Transformers library. Make sure it is installed on your system
from simpletransformers.classification import ClassificationModel
#Initializing our Classification Model
#Repeat this step for various Pre-trained models albert, roberta, distilbert, xlnet etc
model = ClassificationModel('bert', 'bert-base-uncased', num_labels=8, weight=None,
                            args={'overwrite_output_dir': True}, use_cuda=False)

#Training the Model
model.train_model(train) 

#Evaluating the model and storing results into the output folder
from sklearn.metrics import f1_score, accuracy_score
def f1_multiclass(labels, preds):
    return f1_score(labels, preds, average='micro')    
result, model_outputs, wrong_predictions = model.eval_model(eval_df, f1=f1_multiclass, acc=accuracy_score)

#Testing the model by providing manual Text input
predictions, raw_outputs = model.predict('thankyou so much kavita maam')
print(predictions)