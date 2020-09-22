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
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

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
a_embeddings = model.encode(a)
