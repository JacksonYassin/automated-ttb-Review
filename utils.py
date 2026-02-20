import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
import re

##############
#
# Library of commonly used helper functions 
#
##############


def normalize_word(w:str):
    """
    Remove all non-alphanumeric characters
    """
    return re.sub(r"[^a-z0-9]", "", w.lower())


def normalize_for_pattern(w:str):
    """
    Remove trailing whitespace and make lowercase
    """
    return w.strip().lower()


def df_to_list(df:pd.DataFrame):
    """
    Creates a formatted list of tesseract and easyOCR dataframe information
    """
    return [(row["word"], row["location"], row["confidence"]) for _, row in df.iterrows()]


def joined_text(source:tuple):
    """
    Extracts words from output tuples and combines them into larger strings
    """
    if isinstance(source, set):
        return " ".join(w[0] for w in source)
    
    return " ".join(source["word"].astype(str))


def df_to_lookup(df: pd.DataFrame):
    """
    Creates a buffer for efficient lookup in dataframes when finding words
    """
    lookup = {}
    for _, row in df.iterrows():
        key = normalize_word(row["word"])
        if key and key not in lookup:
            lookup[key] = (row["word"], row["location"], row["confidence"])

    return lookup


def find_pattern_in_sources(pattern, fused_words, tess_scan, easy_scan):
    """
    Searches for a regex pattern across all sources, returns first match
    Uses non-fused sources because sometimes tesseract and easyOCR find different things
    """
    for w in fused_words:
        if pattern.search(normalize_for_pattern(w[0])):
            return w
    for _, row in tess_scan.iterrows():
        if pattern.search(normalize_for_pattern(str(row["word"]))):
            return (row["word"], row["location"], row["confidence"])
    for _, row in easy_scan.iterrows():
        if pattern.search(normalize_for_pattern(str(row["word"]))):
            return (row["word"], row["location"], row["confidence"])
        
    return None


def cluster_dbscan(words:set, eps:int=100):
    """
    Clusters spatially aligned words together with DBSCAN
    (i.e. finds paragraphs/text blocks in the label)
    """
    # pull only locations
    locations = np.array([[int(w[1][0]), int(w[1][1])] for w in words])

    labels = DBSCAN(eps=eps, min_samples=1).fit_predict(locations)

    # assign cluster to each word
    clusters = {}
    for label, w in zip(labels, words):
        clusters.setdefault(label, []).append(w)

    return list(clusters.values())
