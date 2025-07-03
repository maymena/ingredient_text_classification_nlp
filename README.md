# My Solution to the Search by Ingredients Challenge

![image](https://github.com/user-attachments/assets/42073ccf-9f8a-4fe1-8e8c-41f6fb51b868)


- Each ingredient string is stripped of whitespace, lowercased, and hyphens are replaced with spaces then parsed with "ingredient_parser" package.
- Each of the remaining words is singularized using the "inflect" package.

## 🥦 Vegan/non-vegan ingredient text classification

- Each word in the ingredient phrase is checked against a comprehensive non-vegan list (extracted from https://github.com/hmontazeri/is-vegan/blob/master/src/i18n/en/canbevegan.json ; I further expanded this list with ChatGPT suggestions and specific food groups I requested).
- If a non-vegan word is found, the surrounding context (word before or after) is checked for plant-based origin:
  - NLTK’s WordNet is used to check for synonyms, hypernyms (is-a), and hyponyms (type-of) of plant-based categories for the word before the non-vegan word.
  - If the context confirms plant-based origin, the ingredient is considered vegan despite the non-vegan word match (e.g., "almond" in "almond milk").
  - Special vegan keywords and substitute indicators are checked, such as "vegan", "non-dairy", "plant based", "tofu", "seitan", "beyond", "impossible", "not", "no", "free", "alternative", and "substitute".
- If all checks pass, the ingredient is considered vegan; otherwise, it is labeled non-vegan.
  
### Suggestion for improving generalization even more:
- Testing whether the nisuga/food_type_classification_model would improve the results.
  - This model was trained using a dataset from USDA FoodData Central which contains the ANIMAL_BASED and PLANT_BASED classification labels based on the available protein type in a food product

## 🥑 Keto/non-keto ingredient text classification

- I downloaded data from Open Food Facts https://world.openfoodfacts.org/data containing ~ 3.8 million product names and carbohydrate values.
    - Preprocessed the data: cleaned, parsed, singularized, deduplicated (reduced it to ~800000 entries)
    - Tokenized and embedded product names using the pre-trained "all-MiniLM-L6-v2" sentence transformer.
    - My preprocessed and embedded file can be downloaded: https://drive.google.com/file/d/1b289yPBgO30k8x1j5KoVH8l2D1MZ3LAD/view?usp=sharing. Size: 1.63GB. This file is required for the code to run.  
- The `is_ingredient_keto` function embeds each ingredient and computes cosine similarity between the ingredient's embedding and all product embeddings from the Open Food Facts dataset.
- The function identifies the product with the highest cosine similarity score.
- The carbohydrate value for the most similar product is retrieved.
- If the carbohydrate value is above 10g/100g, the ingredient is not keto.
- If the best match similarity is below a set threshold (a configurable parameter), the ingredient is not keto.

### Suggestions for further improvements: 
**To improve carbohydrate data and classification speed (dataset improvement):**
- Using CORGIS dataset instead of data from Open Food Facts (https://corgis-edu.github.io/corgis/csv/food/?utm_source=chatgpt.com)
  - Open Food Facts contained ~4M products and after my preprocessing ~800K products. Performing similarity search across 800K product embeddings can be computationally intensive for real-time applications. In contrast, CORGIS has "only" 70K products so it may enable faster similarity matching.While this may reduce coverage, it likely retains the most commonly used products.
  - Open Food Facts may contain inaccurate carbohydrate values, as the data is user-contributed.. In contrast, CORGIS is a curated dataset with vetted information.

**To improve ingredient matching (model improvement):**
- Using a larger sentence transformer model like "all-mpnet-base-v2" or "bge-large-en-v1.5" which are slower but more accurate than "all-MiniLM-L6-v2". A smaller model was initially used to reduce inference time in the deployed application.
- Fine-tuning the selected transformer model on a food-domain dataset may improve ingredient matching accuracy.

## I achieved 99% accuracy on the vegan task and 70% on the keto task, the latter was affected by ground truth keto label incorrectness.

![image](https://github.com/user-attachments/assets/55c20d0a-ef7f-4f53-8084-2057b899f8c3)


### 📁 Data Preprocessing, Embeddings, and Prediction Error Analysis (including inspection of **ground truth keto label inconsistencies**)

**Files in `nb/src/preprocess_error_analysis_and_interim_data`:**
- **keto_error_analysis.ipynb**
  - Analyzes keto prediction errors.
  - Identifies label incorrectness in the ground truth keto dataset.
  - Identifies additional sources of misclassification
    - Fixes some carbohydrate values since the Open Food Facts database is open source that users can contaminate with incorrect data.
      - This was addressed by calculating the median carbohydrate value for duplicate product names.  
- **preprocess_open_food_facts.py**
  - cleans, parses, and singularizes, product names in the Open Food Facts dataset.
  - Removes duplicate product names and prepares data for embedding.
- **embed_food.ipynb**
  - Embeds product names from the Open Food Facts database using a sentence transformer model.
  - Produces the dataset used for ingredient matching.

### Requirements to run the code: 
Before running the code:
  - **Preprocessed Data File:**  
   Download the preprocessed file containing `product_name`, embeddings, and carbohydrate values (size: 1.63GB).  
   Save the file in the `src` folders.  
   > 📦 [Download link (Google Drive)](https://drive.google.com/file/d/1b289yPBgO30k8x1j5KoVH8l2D1MZ3LAD/view?usp=sharing))

  - **Windows Users – Set Docker Memory via `.wslconfig` (minimum 11GB):**  
   If you are using Windows, ensure your Docker (WSL2) VM has enough memory.  
   Check or create the file:
   > C:\Users\<username>\.wslconfig
   Add or update the following content:
   ```ini
    [wsl2]
    memory=11GB


