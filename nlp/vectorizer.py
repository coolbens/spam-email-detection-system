from typing import Iterable, Literal

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Type alias nga nagtugot ra og duha ka valid values:
# "tfidf" ug "bow" (Bag of Words).
# Makatabang ni para sa type checking ug code readability.
VectorizerType = Literal["tfidf", "bow"]


# Muhimo og vectorizer base sa gipili nga klase.
# Parameters:
#   kind         -> "tfidf" o "bow"
#   max_features -> maximum nga number sa words/features nga gamiton
#
# ngram_range=(1, 2) nagpasabot nga:
#   unigram  -> single word ("free")
#   bigram   -> duha ka magkadugtong nga words ("free offer")
#
# Example:
#   build_vectorizer("tfidf")
#   build_vectorizer("bow")
def build_vectorizer(kind: VectorizerType = "tfidf", max_features: int = 5000):

    # Kung Bag of Words ang gipili.
    if kind == "bow":
        return CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2)
        )

    # Default nga TF-IDF vectorizer.
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2)
    )


# I-train (fit) ang vectorizer gamit ang training texts
# ug dayon i-convert ang text ngadto sa numerical vectors.
#
# fit():
#   Makakat-on sa vocabulary gikan sa training data.
#
# transform():
#   Moconvert sa text ngadto sa feature matrix.
#
# fit_transform():
#   Combination sa fit() ug transform().
#
# Example:
#   texts = ["free money", "hello world"]
#   X_train = fit_vectorizer(vectorizer, texts)
def fit_vectorizer(vectorizer, texts: Iterable[str]):

    # Pagkat-on sa vocabulary ug paghimo sa vectors.
    return vectorizer.fit_transform(texts)


# Gamiton ang existing nga trained vectorizer
# aron i-convert ang bag-ong texts ngadto sa vectors.
#
# Dili na siya magkat-on ug bag-ong vocabulary.
# Ang vocabulary nga nakat-onan sa training data
# mao gihapon ang gamiton.
#
# Example:
#   new_texts = ["claim your free prize"]
#   X_test = transform_texts(vectorizer, new_texts)
def transform_texts(vectorizer, texts: Iterable[str]):

    # Convert bag-ong text ngadto sa vectors
    # gamit ang existing vocabulary.
    return vectorizer.transform(texts)