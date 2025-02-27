import re

import pandas as pd
from rapidfuzz import process, fuzz

from utils.variables import COLOURS
from utils.preprocess import longest_common_subsequence


def save_colours_list(colours_list: list, 
                      path: str = "unique_colours.txt",
                      num_unique_colours: int = 500):
    count = 0
    colour: str
    corrected_list: list = []
    with open(path, "w") as f:
        for colour in colours_list:
            f_colour = re.sub(r"[^a-z0-9\s+]", "", colour)
            if len(f_colour) > 2 and colour.isalpha() and count < num_unique_colours:
                corrected_list.append(f_colour)
                f.write(f_colour + "\n")
                count += 1
    return corrected_list


def colour_normalize(colour: str,
                    colours_list: list = COLOURS, 
                    threshold: float = 0.8) -> str:
    processed_colour = colour.lower().strip()
    matched, score, _ = process.extractOne(processed_colour,
                                           colours_list,
                                           scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return matched
    # else:  # too long for diminishing returns
    #     for colour in colours_list:
    #         lcs_length = longest_common_subsequence(processed_colour, colour)
    #         # a ratio to check if its matched is good but with 3-4 char colours, it should be exact or one off. e.g. "red" "ref", "blue" "blur"
    #         if (len(colour) > 4 and lcs_length / len(colour) >= threshold and lcs_length > 1) or \
    #            (len(colour) <= 4 and lcs_length + 1 >= len(colour) and lcs_length > 1):
    #             return colour
    return "unknown"



def colour_match(query: str,  # min 3 chr, txt
                 prod_colour: str, 
                 colours_list: list = COLOURS):
    for colour in colours_list:
        if colour in query:
            return 1.0 if colour in prod_colour else 0.0
    return 0.0
