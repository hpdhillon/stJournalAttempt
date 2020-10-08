import streamlit as st
import nltk as nltk
import math as math
from nltk.tokenize import WordPunctTokenizer
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
sid = SentimentIntensityAnalyzer()
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
from nltk.corpus import wordnet_ic
import scipy
import torch
#import sklearn
from scipy import spatial
from sentence_transformers import SentenceTransformer
@st.cache(allow_output_mutation=True)
def load_my_model():
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    return model

def polarization_heuristic(user_journal):
    # score (0-1) 0 is full helplessness and 1 is super happy
    # get determiners from all submissions
    # find proportion of polarized determiners to all determiners
    # return that minus 1
    # print(user_journal.full_text)
    # tagged_words = nltk.pos_tag(user_journal.full_text.split(' '))
    tagged_words = nltk.pos_tag(WordPunctTokenizer().tokenize(user_journal))
    word_pairs = [(word, nltk.tag.map_tag('en-ptb', 'universal', tag)) for word, tag in tagged_words]
    potential_absolutist_word = []
    for word_tag_pair in word_pairs:
        # word[1] = Part of speech classified
        # RB = Determiners (Some, All, Few, etc., a)
        if (word_tag_pair[1] in ["DET","ADV", "ADJ"]):
            potential_absolutist_word.append(word_tag_pair[0]) # jush push the word, not

    # absolutist ADJ, DET & ADV
    absolutist_words = ["all", "always", "blame", "every", "never", "absolutely", "complete", "completely", "constant", "definetly", "entire", "ever", "full", "totally", "endless"]

    amount_used_in_text = 0
    for word in absolutist_words:
        if word in user_journal:
            amount_used_in_text = amount_used_in_text + 1


    # how many words would be significant (40% of determiners)
    threshold = math.ceil(len(potential_absolutist_word) * 0.40)

    if (threshold == 0):
        return 0 # neutral

    if (amount_used_in_text/threshold > 1):
        return -1;
    else:
        return 1 - (amount_used_in_text/threshold);

#st.title('Hello!')
#st.markdown("![Alt Text](https://data.whicdn.com/images/260389678/original.gif)")
sentence = st.text_area("what's on your mind?")
#button = st.button()
m = sid.polarity_scores(sentence)
score = m['compound']
a = sentence.split('.')
a = a[:len(a)-1]
if len(a) > 2:
    b = a[len(a)-2]+a[len(a)-1]
    c = sid.polarity_scores(b)
    score = c['compound']
d = polarization_heuristic(sentence)
EHS = pd.read_csv("EHS.csv")
sentence_embeddings = EHS.values.tolist()
OPTO = pd.read_csv("OPTO.csv")
optimistic_embeddings = OPTO.values.tolist()
model = load_my_model()
booleon = 0
if st.button('Analysis'):
    a_embeddings = model.encode(a)
    for j in range(len(a_embeddings)):
        for i in range(len(sentence_embeddings)):
            result = 1 - spatial.distance.cosine(sentence_embeddings[i], a_embeddings[j])
            if result > .8:
                booleon = booleon - 1
                #print(a[j])
                #st.write('You sound helpless, this sentence concerned me:', a[j])
                break
    if booleon > 0:
        for j in range(len(a_embeddings)):
            for i in range(len(optimistic_embeddings)):
                result = 1 - spatial.distance.cosine(optimistic_embeddings[i], a_embeddings[j])
                if result > .8:
                    booleon = booleon + 1
    rent = .3*(booleon/len(a_embeddings))
    score = 50 + 50*(rent+(score*.4)+(d*.3))
    #score = 50 + (50*(rent+((score+d-.5)/2)))
    st.write('your score is:', score)
    #st.empty()
    if booleon <  -2:
        st.write("You sound sad. That's fine. Let it all out.")
        st.markdown("![Alt Text](https://media.tenor.com/images/ff4a60a02557236c910f864611271df2/tenor.gif)")
    if boolean > 2:
        st.write("You are a ray of sunshine today! Keep it up playa!")
        st.markdown("![Alt Text](https://media.tenor.com/images/2aa9b6f3a7d832c2ff1c1a406d5eae73/tenor.gif)")
if st.button('Save as text file'):
    import numpy as np
    import base64
    df = pd.DataFrame(a)
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown('### **⬇️ download output txt file **')
    href = f'<a href="data:file/csv;base64,{b64}">download txt file</a> (right-click and save as ".txt")'
    st.markdown(href, unsafe_allow_html=True)
