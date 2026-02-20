import pytesseract
import easyocr
import cv2
import re
import pandas as pd
import matplotlib.pyplot as plt
from rapidfuzz import fuzz
from parsing import *


##############
#
# Functions to extract and process text from label images
#
##############


def ocr_scan(image:str, easy_reader:easyocr.Reader):
    """
    Main function to extract text from label images

    Uses two different OCR methods: tesseract and easyOCR
    Takes outputs from these two different methods and creates a dataframe

    Returns two dataframes with columns "word", "location", and "confidence"
    """

    img = cv2.imread(image)

    # get scan from tesseract
    tess = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    tess_df = pd.DataFrame(tess)

    # remove non-words
    tess_df = tess_df[tess_df["text"].astype(str).str.strip() != ""]

    # extract word, location, confidence score
    tess_out = pd.DataFrame({
        "word": tess_df["text"],
        "location": list(zip(tess_df["left"], tess_df["top"])),
        "confidence": tess_df["conf"].astype(float)
    })

    # easyOCR sometimes spaces out words (e.g. "C O M P A N Y")
    # use regex to correct for this error
    # if the pattern "letter space letter space" occurs in word, it is likely a spaced out word
    spaced_letter_pattern = re.compile(r'[A-Za-z] [A-Za-z] ')

    # get scan from easyOCR and extract individual words
    easy = easy_reader.readtext(img)
    # extract word, location, confidence score
    easy_rows = []
    for box, text, conf in easy:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        
        # check if text contains spaced-out letters pattern
        if spaced_letter_pattern.search(text):
            # keep the text as-is
            easy_rows.append({
                "word": text,
                "location": (min(xs), min(ys)),
                "confidence": float(conf)
            })
        else:
            # split text into individual words
            for word in text.split():
                easy_rows.append({
                    "word": word,
                    "location": (min(xs), min(ys)),
                    "confidence": float(conf)
                })
            
    easy_out = pd.DataFrame(easy_rows)

    # sort dataframes from top left to bottom right, line by line (e.g. reading order)
    tess_out = (
        tess_out.assign(x=lambda df: df["location"].map(lambda loc: loc[0]),
                        y=lambda df: df["location"].map(lambda loc: loc[1]),
                        )
        .sort_values(["y", "x"])
        .drop(columns=["x", "y"])
        .reset_index(drop=True)
    )

    easy_out = (
        easy_out.assign(x=lambda df: df["location"].map(lambda loc: loc[0]),
                        y=lambda df: df["location"].map(lambda loc: loc[1]),
                        )
        .sort_values(["y", "x"])
        .drop(columns=["x", "y"])
        .reset_index(drop=True)
    )

    return tess_out, easy_out


def make_fusion_list(tess_pd:pd.DataFrame, easy_pd:pd.DataFrame):
    """
    Uses tesseract and easyOCR scans to improve OCR results

    EasyOCR and tesseract use different methods to detect text and sometimes disagree about words
    Their different scanning techniques cause them to have errors in different places,
    meaning they are complementary systems

    This function pulls the same words from each scan and puts them in the same list for later analysis

    returns tuple in format ((word 1, word 2), (loc 1, loc 2), (confidence 1, confidence 2))
    """
    
    
    similar_list = []

    # there will sometimes be dissagreements about the number of words between methods
    # determine which list is shorter so we can iterate over words & fuse without stepping out of index range
    if len(tess_pd) <= len(easy_pd):
        short_df = tess_pd
        long_df = easy_pd
        short_name = "tess"
    else:
        short_df = easy_pd
        long_df = tess_pd
        short_name = "easy"

    len_short = len(short_df)
    len_long = len(long_df)

    for row in range(len_short):
        for i in range(len_long):
            # get words, locations, and confidence scores from both dfs
            short_word = short_df.iloc[row, 0]
            long_word = long_df.iloc[i, 0]

            short_loc = short_df.iloc[row, 1]
            long_loc = long_df.iloc[i, 1]

            short_conf = short_df.iloc[row, 2]
            long_conf = long_df.iloc[i, 2]

            # if the words are similar enough and likely on the same line, they're likely the same word
            if (
                fuzz.ratio(short_word, long_word) >= 60
                and (abs(long_loc[1] - short_loc[1]) < 25)
            ):
                # keep constant ordering in output 
                if short_name == "tess":
                    tess_word, easy_word = short_word, long_word
                    tess_loc, easy_loc = short_loc, long_loc
                    tess_conf, easy_conf = short_conf, long_conf
                else:
                    tess_word, easy_word = long_word, short_word
                    tess_loc, easy_loc = long_loc, short_loc
                    tess_conf, easy_conf = long_conf, short_conf

                # tuple -> ((word 1, word 2), (loc 1, loc 2), (confidence 1, confidence 2))
                similar_list.append(
                    (
                        (tess_word, easy_word),
                        (tess_loc, easy_loc),
                        (tess_conf, easy_conf),
                    )
                )

    return similar_list


def fuse_lists(similar_list:list):
    """
    Creates high-quality scan list using earlier fused list

    Only one of the two words is selected for the return list
    Words are selected based on the models' confidence (higher is better)

    returns list of (word, location, confidence)
    """
    fused_set = set() # use set to prevent duplicate entries

    for matched_word in similar_list:
        # use the word with a higher associated confidence score
        if matched_word[2][0] >= matched_word[2][1]:
            fused_set.add((matched_word[0][0], matched_word[1][1], matched_word[2][0]))
        else:
            fused_set.add((matched_word[0][1], matched_word[1][1], matched_word[2][1]))

    return fused_set
