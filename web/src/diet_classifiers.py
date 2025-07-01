import json
import sys
from argparse import ArgumentParser
from typing import List
from time import time
import pandas as pd
try:
    from sklearn.metrics import classification_report
except ImportError:
    # sklearn is optional
    def classification_report(y, y_pred):
        print("sklearn is not installed, skipping classification report")


def is_ingredient_keto(ingredient: str) -> bool:
    # TODO: Implement (Copy your solution from `nb/src/diet_classifiers.py`)
    return False


def is_ingredient_vegan(ingredient: str) -> bool:
    # TODO: Implement (Copy your solution from `nb/src/diet_classifiers.py`)
    return False


def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, ingredients))


def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))
