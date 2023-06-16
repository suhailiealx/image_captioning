import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from keras.models import Model, load_model
from pickle import load
from keras.applications.xception import Xception #to get pre-trained model Xception
from keras_preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

import math
import nltk
from nltk import tokenize
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help="Image Path")
args = vars(ap.parse_args())
img_path = int(args['image'])

dataset_text = "C:\\Users\\asus\\Downloads\\Flickr8k_text"
dataset_images = "C:\\Users\\asus\\Downloads\\Flickr8k_Dataset\\Flicker8k_Dataset"
filename = dataset_text + "/" + "Flickr_8k.testImages.txt"

stop_words = set(stopwords.words('english'))

def load_fp(filename):
       # Open file to read
   file = open(filename, 'r')
   text = file.read()
   file.close()
   return text

def load_photos(filename):
    file = load_fp(filename)
    photos = file.split("\n")[:-1]
    return photos

def load_clean_descriptions(filename, photos):
    #loading clean_descriptions
    file = load_fp(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
                desc = 'startseq ' + " ".join(image_caption) + ' endseq'
                descriptions[image].append(desc)

    return descriptions

def load_all_descriptions(filename, photos):
    #loading clean_descriptions
    file = load_fp(filename)
    descriptions = {}
    for line in file.split("\n"):
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
                desc = 'startseq ' + " ".join(image_caption) + ' endseq'
                descriptions[image].append(desc)
            else:
                desc = 'startseq ' + " ".join(image_caption) + ' endseq'
                descriptions[image].append(desc)

    return descriptions

def load_features(photos):
    #loading all features
    all_features = load(open("features.p","rb"))

    #selecting only needed features
    features = {k:all_features[k] for k in photos}

    return features

def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        print("ERROR")
    else:
        print("Image successfully opened")
        image = image.resize((299,299))
        image = np.array(image)
        # for 4 channels images, we need to convert them into 3 channels
        if image.shape[2] == 4:
            image = image[..., :3]
    
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        #print("sequence = tokenizer.texts_to_sequences([in_text])[0] : ",sequence)
        sequence = pad_sequences([sequence], maxlen=max_length)
        #print("sequence = pad_sequences([sequence], maxlen=max_length) : ",sequence)
        pred = model.predict([photo,sequence], verbose=0)
        #print("pred = model.predict([photo,sequence], verbose=0) : ",pred)
        pred = np.argmax(pred)
        #print("pred = np.argmax(pred) : ", pred)
        word = word_for_id(pred, tokenizer)
        #print("word = word_for_id(pred, tokenizer) : ", word)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    
    return in_text

def clean_description(description) :
    return (description.replace('startseq', '').replace('endseq', ''))

def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

def extract_keyword(sentence) :
    total_words = sentence.split()
    total_word_length = len(total_words)
    
    total_sentences = tokenize.sent_tokenize(sentence)
    total_sent_len = len(total_sentences)

    tf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1

    # Dividing by total_word_length for each dictionary element
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())

    idf_score = {}
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1

    # Performing a log and divide
    idf_score.update((x, math.log(int(total_sent_len)/y)) for x, y in idf_score.items())

    tf_idf_score = {key: tf_score[key] * idf_score.get(key, 0) for key in tf_score.keys()}

    result = dict(sorted(tf_idf_score.items(), key = itemgetter(1), reverse = True)[:5]) 
    
    return result


# ------- MAIN -------
test_imgs = load_photos(filename)
test_descriptions = load_clean_descriptions("descriptions.txt", test_imgs)
test_descriptions_all = load_all_descriptions("descriptions.txt", test_imgs)
test_features = load_features(test_imgs)

max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models_simple/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")
file = dataset_images + "/" + test_imgs[img_path]
photo = extract_features(file, xception_model)
img = Image.open(file)
description = generate_desc(model, tokenizer, photo, max_length)
print("nn\n")
print("generated_description : ", clean_description(description))
cand = [x for x in description.split()
        if ('startseq' not in x and 'endseq' not in x)]
print("\nground_t_description : ")
ref = []
for i in range(len(test_descriptions_all[test_imgs[img_path]])) :
    print(clean_description(test_descriptions_all[test_imgs[img_path]][i]))
    ref.append([x for x in test_descriptions_all[test_imgs[img_path]][i].split()
                if ('startseq' not in x and 'endseq' not in x)])

#bleu = sentence_bleu(ref, cand, weights=(1,0,0,0))
#print("bleu_score : ", bleu)

print("\nkeywords : ", extract_keyword(clean_description(description)))

im = plt.imread(dataset_images + '/' + test_imgs[img_path])
plt.imshow(im)
plt.show()

#bleu_score = 0.0

# for i in range(len(test_imgs)) :
#     file = dataset_images + "/" + test_imgs[i]
#     photo = extract_features(file, xception_model)
#     description = generate_desc(model, tokenizer, photo, max_length)
#     cand = [x for x in description.split()
#            if ('startseq' not in x and 'endseq' not in x)]
#     ref = [x for x in test_descriptions[test_imgs[i]][0].split()
#             if ('startseq' not in x and 'endseq' not in x)]
    
#     print(ref, cand)

    #score = sentence_bleu(ref,cand)
    #print(score)
    #bleu_score += score
    #print(bleu_score)

#print(bleu_score/len(test_imgs))
