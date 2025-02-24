ROOT_FOLDER = "formatted_esci/"

COLUMNS_TO_PROCESS =  ["query", 
                       "product_title", 
                       "product_description",
                       "product_brand",
                       "product_color"]  # product_text is already a combined version


# XXX: use smoothed labels?, use scaled scoring?
SCORE_MAP = {
    "Exact": 3,  # exact
    "Substitute": 2,  # substitute
    "Complement": 1,  # complementary
    "Irrelevant": 0   # irrelevant
    }

