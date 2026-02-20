import pandas as pd
import re
from utils import *

##############
#
# Functions used to determine if the appropriate application elements are present
#
##############


NET_UNITS = [
    "fl", "fl.", "oz", "oz.", "floz",
    "pint", "pt", "pt.",
    "quart", "qt", "qt.",
    "gallon", "gal", "gal."
]


def is_net_content_token(word:str):
    """
    Checks if there is number and measurement in the same word
    """
    w = word.lower()
    has_number = bool(re.search(r"\d", w))
    has_unit = any(unit in w for unit in NET_UNITS)
    
    return has_number and has_unit


def find_net_content_in_cluster(clusters, dist_thresh=60):
    """
    Checks if there is number and measurement nearby
    """
    for cluster in clusters:
        words = [(w[0], w[1]) for w in cluster]  # (word, (x,y))

        # do single-token check 
        for word, _ in words:
            if is_net_content_token(word):
                return (True, cluster)

        # use regex to verify number/unit tokens, then locations to verify they're proximate
        numbers = [(w, loc) for w, loc in words if re.search(r"\d", w)]
        units = [(w, loc) for w, loc in words if any(u in w.lower() for u in NET_UNITS)]

        for (_, loc1) in numbers:
            for (_, loc2) in units:
                if abs(loc1[0] - loc2[0]) < dist_thresh and abs(loc1[1] - loc2[1]) < dist_thresh:
                    return (True, cluster)

    return (False, None)


def find_elements(
    app_info:list[str],
    fused_words:set,
    tess_scan:pd.DataFrame,
    easy_scan:pd.DataFrame
):
    """
    Finds required elements in the image of the label

    1. For each element in the application, search the fused word list first (more accurate)
    2a. If it cannot be found in the fusion word lists, search the Tesseract and EasyOCR scans indidivually
    2b. If it cannot be found in any of the lists, append None to the return list and move on
    3. If it can, append the word and its location to the return list

    returns list of info in order:
    [brand name, class, fanciful name, bottler name, bottler address, alcohol content, net content]
    """

    # normalize words for lookup in the fused list
    fused_lookup = {
        normalize_word(w[0]): w
        for w in fused_words
        if normalize_word(w[0])
    }
    
    tess_lookup = df_to_lookup(tess_scan)
    easy_lookup = df_to_lookup(easy_scan)

    # main search loop
    # compares stored application data to scanned info
    results = []
    for phrase in app_info:
        if phrase is None:
            results.append(None)
            continue
        
        # break database info into tokens (e.g. "Brewing Company" --> "Brewing", "Company")
        words = phrase.split()
        matches = []
        missing = False
        for word in words:
            # normalize and search found words
            key = normalize_word(word)
            found = (
                fused_lookup.get(key)
                or tess_lookup.get(key)
                or easy_lookup.get(key)
            )
            if found is None:
                missing = True
                continue

            matches.append(found)
            
        results.append(None if missing else matches)

    # search for alcohol content (format = number + %)
    alcohol_pattern = re.compile(r"\d+(\.\d+)?%")
    results.append(find_pattern_in_sources(alcohol_pattern, fused_words, tess_scan, easy_scan))

    # search for net contents (u.s. standard measurements)
    net_pattern = re.compile(
        r"\b\d+(\.\d+)?\s*("
        r"fl\.?\s*oz\.?|"
        r"pt\.?|pint(s)?|"
        r"qt\.?|quart(s)?|"
        r"gal\.?|gallon(s)?"
        r")\b",
        re.IGNORECASE
    )
    
    # join found wordes together into output list
    for source in (fused_words, tess_scan, easy_scan):
        text = joined_text(source)
        match = net_pattern.search(text)
        if match:
            results.append(match.group())
            break
    else:
        results.append(None)

    return results


def verify_locations(
    found_elements:list,
    fused_words:set,
    tess_scan:pd.DataFrame,
    easy_scan:pd.DataFrame,
    eps:int=100
):
    """
    Verify that alochol percentage and net content are associated with numerical value
    """
    # create clusters of spatially alligned words (e.g. paragraphs) using DBSCAN 
    fused_clusters = cluster_dbscan(list(fused_words), eps) if fused_words else []
    tess_clusters = cluster_dbscan(df_to_list(tess_scan), eps) if len(tess_scan) > 0 else []
    easy_clusters = cluster_dbscan(df_to_list(easy_scan), eps) if len(easy_scan) > 0 else []

    all_clusters = [fused_clusters, tess_clusters, easy_clusters]
    results = []
    
    # alcohol content and net contents are last two elements in previous search output list
    for i in range(len(found_elements) - 2, len(found_elements)):
        element = found_elements[i]
        
        # element not found in search step
        if element is None:
            results.append((False, None))
            continue
        
        # enforce standard output formatting (list of tuples)
        if isinstance(element, tuple):
            element = [element]
        
        # verify alcohol content 
        if i == len(found_elements) - 2:
            found = (False, None)
            for clusters in all_clusters:
                found = find_alcohol_statement_in_cluster(clusters)
                if found[0]:
                    break
            results.append(found)
            continue
            
        # verify net contents
        if i == len(found_elements) - 1:
            found = (False, None)
            for clusters in all_clusters:
                found = find_net_content_in_cluster(clusters)
                if found[0]:
                    break
            results.append(found)
            continue
            
    return results


def find_words_in_cluster(words:list[str], clusters:list[list]):
    """
    Verify that found phrases are all in the same cluster
    """
    for cluster in clusters:
        cluster_words = {normalize_word(w[0]) for w in cluster}
        if all(normalize_word(word) in cluster_words for word in words):
            return (True, cluster[0][1])
        
    return (False, None)


def find_alcohol_statement_in_cluster(clusters:list[list]):
    """
    Uses regex to verify if a valid alcohol content statement can be found in a cluster
    Patterns checked for are:
    Valid patterns require:
    - "alcohol" or "alc" (with optional period)
    - "volume" or "vol" (with optional period)
    - a percentage (number with %)
    
    Robust to different tokenizations (e.g. "alcohol by volume" and "ALC/VOL")

    returns (True, (location)) if found, (False, None) otherwise
    """
    
    alcohol_pattern = re.compile(r"alc(ohol)?\.?")
    volume_pattern = re.compile(r"vol(ume)?\.?")
    percent_pattern = re.compile(r"\d+\.?\d*%")
    
    for cluster in clusters:
        has_alcohol = False
        has_volume = False
        has_percent = False
        
        for w in cluster:
            word = normalize_for_pattern(w[0])
            if alcohol_pattern.search(word):
                has_alcohol = True
            if volume_pattern.search(word):
                has_volume = True
            if percent_pattern.search(word):
                has_percent = True
        
        if has_alcohol and has_volume and has_percent:
            return (True, cluster[0][1])
    
    return (False, None)


def find_number_in_cluster(target_word:str, clusters:list[list]):
    """
    Verify that a number exists in the same paragraph as the target word
    """
    number_pattern = re.compile(r"\d+\.?\d*")
    for cluster in clusters:
        cluster_words = [normalize_word(w[0]) for w in cluster]
        if normalize_word(target_word) in cluster_words:
            for w in cluster:
                if number_pattern.search(normalize_word(w[0])):
                    return (True, cluster[0][1])
                
    return (False, None)


def verify_government_warning(
    fused_words:set,
    tess_scan:pd.DataFrame,
    easy_scan:pd.DataFrame
):
    """
    Verifies that the government warning label appears correctly on the label

    Checks:
    - "GOVERNMENT" and "WARNING" are be capitalized
    - "Surgeon" and "General" have capital S and G
    - Matches wording and punctuation
    
    returns (True, (location)) if valid warning found, (False, None) otherwise
    """
    required_words = [
        "GOVERNMENT", "WARNING:", "(1)", "According", "to", "the", "Surgeon",
        "General,", "women", "should", "not", "drink", "alcoholic", "beverages",
        "during", "pregnancy", "because", "of", "the", "risk", "of", "birth",
        "defects.", "(2)", "Consumption", "of", "alcoholic", "beverages",
        "impairs", "your", "ability", "to", "drive", "a", "car", "or",
        "operate", "machinery,", "and", "may", "cause", "health", "problems."
    ]
    
    # collect all available words from all sources
    available_words = []
    first_location = None
    
    for w in fused_words:
        available_words.append(w[0])
        if first_location is None:
            first_location = w[1]
    
    for _, row in tess_scan.iterrows():
        available_words.append(row["word"])
        if first_location is None:
            first_location = row["location"]
    
    for _, row in easy_scan.iterrows():
        available_words.append(row["word"])
        if first_location is None:
            first_location = row["location"]
    
    # check if all required words are present
    if contains_all_words(required_words, available_words):
        return (True, first_location)
    
    return (False, None)


def contains_all_words(required:list, available:list):
    """
    Checks if all required words can be found in available words
    Handles duplicates by counting occurances
    Unlike other regex checks I use, these must be exact (case sensitive, punctuation, etc)
    """

    required_counts = {}
    for word in required:
        required_counts[word] = required_counts.get(word, 0) + 1
    
    available_counts = {}
    for word in available:
        available_counts[word] = available_counts.get(word, 0) + 1
    
    for word, count in required_counts.items():
        if available_counts.get(word, 0) < count:
            return False
    
    return True