import numpy as np;
import nltk;
import random;
import pickle;
from nltk.tokenize import word_tokenize;
from nltk.stem import WordNetLemmatizer;
from collections import Counter;

lemmatizer = WordNetLemmatizer();
n_lines = 1000000;


def create_lexicon(pos, neg):
    lexicon = [];

    for i in [pos, neg]:
        with open(i, "r") as fl:
            content = fl.readlines();
            for word_line in content[:n_lines]:
                token = word_tokenize(word_line.lower());
                lexicon += list(token);

        lexicon = [lemmatizer.lemmatize(i) for i in lexicon];
        cnt = Counter(lexicon);
        lx = [];
        for i in cnt:
            if 2000 > cnt[i] > 30:
                lx.append(i);

        print(len(lx));
        return lx;


def create_features(sample, lex, label):
    features = [];

    with open(sample, "r") as fl:
        content = fl.readlines();
        for word_line in content[:n_lines]:
            token = word_tokenize(word_line.lower());
            token = [lemmatizer.lemmatize(i) for i in token];
            feature = np.zeros(len(lex));
            for word in token:
                if word.lower() in lex:
                    index = lex.index(word.lower());
                    feature[index] += 1;

            feature = list(feature);
            features.append([feature, label]);


    return features;


def create_set(sample1, sample2, tes=0.1) :
    lexicon = create_lexicon(sample1, sample2);
    features = [];
    features += create_features(sample1, lexicon, [1, 0]);
    features += create_features(sample2, lexicon, [0, 1]);
    random.shuffle(features);

    features = np.array(features);
    test_size = int(tes * len(features));
    train_x = list(features[:, 0][:-test_size]);
    train_y = list(features[:, 1][:-test_size]);

    test_x = list(features[:, 0][-test_size:]);
    test_y = list(features[:, 1][-test_size:]);

    return train_x, train_y, test_x, test_y;

if __name__ == "__main__" :
    train_x, train_y, test_x, test_y = create_set("pos.txt", "neg.txt");
    with open("set.pickle", "wb") as fl :
        pickle.dump([train_x, train_y, test_x, test_y], fl);