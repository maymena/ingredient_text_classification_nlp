import os
import numpy as np
import pandas as pd
import inflect
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from ingredient_parser import parse_ingredient
from diet_constants import non_vegan_ingredients

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



# === GLOBAL SETUP ===
THRESHOLD = 0.009
p = inflect.engine()

# Updated to use the correct Docker-mounted path
DATA_PATH = "/app/web/full_open_food_fact_cleaned_parsed_embedded.parquet"

# Load food data
df_food = pd.read_parquet(DATA_PATH, columns=["product_name", "carbohydrates_100g", "embedding"])
product_embeddings = np.vstack(df_food["embedding"].to_list())
carb_values = df_food["carbohydrates_100g"].to_numpy()
product_names = df_food["product_name"].to_numpy()

# Load sentence embedding model once
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")

# === HELPERS ===
def singularize(text: str) -> str:
    words = text.lower().strip().replace('-', ' ').split()
    singularized = [p.singular_noun(w) if p.singular_noun(w) else w for w in words]
    return ' '.join(singularized)

def is_plant_based_wordnet(phrase: str, i: int) -> bool:
    plant_keywords = ['nut', 'seed', 'almond', 'legume', 'grain', 'fruit', 'vegetable', 'herb', 'alga', 'plant', 'fungus']
    words = phrase.lower().split()

    if i - 1 >= 0:
        word_before = words[i - 1]
        synsets = wn.synsets(word_before, pos=wn.NOUN)

        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.name().lower() in plant_keywords:
                    return True
            for hyper in syn.closure(lambda s: s.hypernyms()):
                if any(plant in hyper.name().lower() for plant in plant_keywords):
                    return True
            for plant in plant_keywords:
                plant_synsets = wn.synsets(plant, pos=wn.NOUN)
                for plant_syn in plant_synsets:
                    for hypo in plant_syn.closure(lambda s: s.hyponyms()):
                        if syn == hypo:
                            return True
        if word_before in ['vegan', 'non-dairy', 'plant based', 'tofu', 'natto', 'seitan', 'silan', 'beyond', 'impossible', 'nori', 'chia', 'portobello', 'champignon', 'not', 'no']:
            return True
        elif i - 2 >= 0:
            two_word_before = words[i - 2] + words[i - 1]
            if two_word_before == 'non dairy':
                return True
            elif i + 1 < len(words):
                word_after = words[i + 1]
                if word_after in ['free', 'alternative', 'substitute']:
                    return True
    return False

# === MAIN FLASK-FACING FUNCTIONS ===
def is_ingredient_keto(ingredient: str) -> bool:
    print("@@@@@ print @@@@@@@@")
    ingredient_str = singularize(ingredient)
    ing_vec = minilm_model.encode([ingredient_str])[0]

    dot_products = np.dot(product_embeddings, ing_vec)
    norms = np.linalg.norm(product_embeddings, axis=1) * np.linalg.norm(ing_vec)
    similarities = dot_products / (norms + 1e-8)

    best_idx = np.argmax(similarities)
    best_sim = similarities[best_idx]
    carbs = carb_values[best_idx]

    if carbs is None or best_sim < THRESHOLD or carbs > 10:
        return False
    return True

def is_ingredient_vegan(ingredient: str) -> bool:
    ingredient_str = singularize(ingredient)
    non_vegan_set = set((v.replace('-', ' ')).lower() for v in non_vegan_ingredients)

    if ingredient_str in non_vegan_set:
        return False

    ingredient_words = ingredient_str.split()
    for i, word in enumerate(ingredient_words):
        if word in non_vegan_set:
            return is_plant_based_wordnet(ingredient_str, i)
    return True


def is_keto(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_keto, ingredients))


def is_vegan(ingredients: List[str]) -> bool:
    return all(map(is_ingredient_vegan, ingredients))
