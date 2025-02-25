""" 
to run: 
python -m unittest tests.test_levenshtein 
"""
import unittest
import time

from abydos.distance import Levenshtein
from utils.preprocess import levenshtein
from rapidfuzz.distance import Levenshtein as rapid_lev


lev1 = levenshtein              # 200x
lev2 = rapid_lev.distance       # 1x
lev3 = Levenshtein().dist_abs   # 300x

class TestLevenshteinFunctions(unittest.TestCase):
    def test_identical_strings(self):
        self.assertEqual(lev1("hello", "hello"), lev2("hello", "hello"))
    
    def test_completely_different_strings(self):
        self.assertEqual(lev1("hello", "world"), lev2("hello", "world"))
    
    def test_empty_string_and_non_empty(self):
        self.assertEqual(lev1("", "world"), lev2("", "world"))
    
    def test_both_empty_strings(self):
        self.assertEqual(lev1("", ""), lev2("", ""))
    
    def test_strings_with_spaces(self):
        self.assertEqual(lev1("hello world", "hello world"), lev2("hello world", "hello world"))
        self.assertEqual(lev1("hello world", "helloworld"), lev2("hello world", "helloworld"))
    
    def test_strings_with_special_characters(self):
        self.assertEqual(lev1("hello!", "hello"), lev2("hello!", "hello"))
        self.assertEqual(lev1("abc#123", "abc123"), lev2("abc#123", "abc123"))
    
    def test_case_sensitivity(self):
        self.assertEqual(lev1("hello", "HELLO"), lev2("hello", "HELLO"))
        self.assertEqual(lev1("abc", "ABC"), lev2("abc", "ABC"))
    
    def test_random_strings_with_differences(self):
        self.assertEqual(lev1("kitten", "sitting"), lev2("kitten", "sitting"))
        self.assertEqual(lev1("flaw", "lawn"), lev2("flaw", "lawn"))
    
    def test_long_strings(self):
        self.assertEqual(lev1("a" * 1000, "b" * 1000), lev2("a" * 1000, "b" * 1000))
        self.assertEqual(lev1("a" * 1000 + "b", "a" * 999 + "c"), lev2("a" * 1000 + "b", "a" * 999 + "c"))

    def test_performance(self):
        iterations = 10000  # Adjust this value if needed
        test_input = ("kitten", "sitting")
        
        # Time lev1
        start_time = time.perf_counter()
        for _ in range(iterations):
            lev1(*test_input)
        time_lev1 = time.perf_counter() - start_time
        
        # Time lev2
        start_time = time.perf_counter()
        for _ in range(iterations):
            lev2(*test_input)
        time_lev2 = time.perf_counter() - start_time
        
        print(f"\nlev1 took {time_lev1:.6f} seconds for {iterations} iterations")
        print(f"lev2 took {time_lev2:.6f} seconds for {iterations} iterations\n")


if __name__ == "__main__":
    unittest.main()
