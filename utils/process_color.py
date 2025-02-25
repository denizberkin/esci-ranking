import pandas as pd
from rapidfuzz import process, fuzz

from utils.variables import COLOURS


def save_colours_list(colours_list: list, path: str = "unique_colours.txt"):
    with open(path, "w") as f:
        for colour in colours_list:
            f.write(colour + "\n")


def colour_normalize(colour: str,
                    colours_list: list = COLOURS, 
                    threshold: float = 0.8) -> str:
    processed_colour = colour.lower().strip()
    matched, score, _ = process.extractOne(processed_colour,
                                           COLOURS,
                                           scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return matched
    else: return "unknown"



def colour_match(query: str,
                 prod_colour: str, 
                 colour_list: list = COLOURS):
    print("Q:",query, "\tPRODCLR: ", prod_colour, end="\t")
    for colour in colour_list:
        if colour is not "" and colour in query:
            print("FOUNDCLR: \t", 1.0 if colour in prod_colour.lower() else 0.0)
            return 1.0 if colour in prod_colour.lower() else 0.0
    return 0.0
