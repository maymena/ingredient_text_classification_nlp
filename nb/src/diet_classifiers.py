import os
import sys
import json
import numpy as np
import pandas as pd
import inflect
import traceback
import re
from time import time
from argparse import ArgumentParser
from typing import List
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
from ingredient_parser import parse_ingredient

# Try importing optional stuff
try:
    from sklearn.metrics import classification_report
except ImportError:
    def classification_report(y, y_pred):
        print("sklearn not installed")

try:
    nltk_path = os.path.expanduser('~/nltk_data')
    wn.synsets('test')  # trigger wordnet
except LookupError:
    import nltk
    nltk.download('wordnet')

import contextlib

# === CONSTANTS ===
THRESHOLD = 0.1
DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), 'full_open_food_fact_cleaned_parsed_embedded.parquet'))
p = inflect.engine()

# === GLOBALS ===
minilm_model = SentenceTransformer("all-MiniLM-L6-v2")

df_food = pd.read_parquet(DATA_PATH, columns=["product_name", "carbohydrates_100g", "embedding"])
product_embeddings = np.vstack(df_food["embedding"].to_list())
carb_values = df_food["carbohydrates_100g"].to_numpy()
product_names = df_food["product_name"].to_numpy()

from diet_constants import (
    non_vegan_ingredients,
    food_measurements_description,
    food_categories,
    non_vegan_categories
)

# using FAISS library
import faiss

# Build the FAISS index ONCE
if product_embeddings.dtype != np.float32:
    product_embeddings = product_embeddings.astype(np.float32)
faiss.normalize_L2(product_embeddings)

# IndexHNSWFlat (Hierarchical Navigable Small World graph)
d = product_embeddings.shape[1]
M = 32  # Number of neighbors in the graph; default is 32, higher = more accuracy

index = faiss.IndexHNSWFlat(d, M)
index.hnsw.efSearch = 55   # Size of dynamic candidate list during search (higher = more accurate, slower)
index.add(product_embeddings)

# === UTILS ===
def singularize(text: str) -> str:
    words = text.lower().strip().replace('-', ' ').split()
    return ' '.join([p.singular_noun(w) if p.singular_noun(w) else w for w in words])

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

# === CLASSIFIERS ===
def is_ingredient_keto(ingredient: str) -> bool:

    try:
        parsed=parse_ingredient(ingredient, separate_names=True) # separate_names=False, foundation_foods=True
        name_texts = [n.text for n in parsed.name]
    except Exception as e:
        name_texts = ingredient
    
    if isinstance(name_texts, list):
        ing = ' '.join(name_texts)
    else:
        ing = name_texts
    
    ingredient_str = singularize(ing)
    ing_vec = minilm_model.encode([ingredient_str])[0].astype(np.float32)
    faiss.normalize_L2(ing_vec.reshape(1, -1))
    
    D, I = index.search(ing_vec.reshape(1, -1), 1)
    best_idx = I[0][0]
    best_sim = D[0][0]
    
    carbs = carb_values[best_idx]

    return carbs is not None and best_sim >= THRESHOLD and carbs <= 10

def is_ingredient_vegan(ingredient: str) -> bool:

    try:
        parsed=parse_ingredient(ingredient, separate_names=True) # separate_names=False, foundation_foods=True
        name_texts = [n.text for n in parsed.name]
    except Exception as e:
        name_texts = ingredient
    
    if isinstance(name_texts, list):
        ing = ' '.join(name_texts)
    else:
        ing = name_texts
    
    ingredient_str = singularize(ing)
    non_vegan_set = set((v.replace('-', ' ')).lower() for v in non_vegan_ingredients)

    if ingredient_str in non_vegan_set:
        return False

    ingredient_words = ingredient_str.split()
    for i, word in enumerate(ingredient_words):
        if word in non_vegan_set:
            return is_plant_based_wordnet(ingredient_str, i)
    return True

def is_keto(ingredients: List[str]) -> bool:
    return all(is_ingredient_keto(ing) for ing in ingredients)

def is_vegan(ingredients: List[str]) -> bool:
    return all(is_ingredient_vegan(ing) for ing in ingredients)

# === MAIN for local testing ===
def main(args):
    ground_truth = pd.read_csv(args.ground_truth, index_col=None)
    ground_truth.to_csv('ground_truth_saved.csv', index=False)
    try:
        start_time = time()
        ground_truth['keto_pred'] = ground_truth['ingredients'].apply(
            lambda s: is_keto(re.findall(r"'(.*?)'", re.sub(r"[\[\]]+", "", s)))
        )
        ground_truth['vegan_pred'] = ground_truth['ingredients'].apply(
            lambda s: is_vegan(re.findall(r"'(.*?)'", re.sub(r"[\[\]]+", "", s)))
        )
        end_time = time()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return -1

    print("THRESHOLD:", THRESHOLD)
    print("=== Keto ===")
    print(classification_report(ground_truth['keto'], ground_truth['keto_pred']))
    print("=== Vegan ===")
    print(classification_report(ground_truth['vegan'], ground_truth['vegan_pred']))
    print(f"== Time taken: {end_time - start_time:.2f} seconds ==")
    
    return 0

# === Notebook usage ===
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ground_truth", type=str, default="/usr/src/data/ground_truth_sample.csv")
    sys.exit(main(parser.parse_args()))
