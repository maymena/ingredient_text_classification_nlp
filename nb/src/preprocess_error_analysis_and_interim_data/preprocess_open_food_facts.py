import os
import unicodedata
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, trim, percentile_approx, udf, when, broadcast, size, regexp_replace, lower, pandas_udf
from pyspark.sql.types import StringType, DoubleType, FloatType, ArrayType
import inflect
import pandas as pd

spark = (
    SparkSession.builder
    .appName("KetoChecker")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "5g")
    .getOrCreate()
)

DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), 'food.parquet'))
ORIG_DIR = os.path.dirname(DATA_PATH)
OUT_PATH = os.path.join(ORIG_DIR, "df_food_parsed.parquet")

print("Current working directory:", os.getcwd())
DATA_PATH = os.path.abspath(os.path.join(os.getcwd(), 'food.parquet'))

df_food = spark.read.parquet(DATA_PATH).select("product_name", "nutriments")
print("Open food fact data loaded")
print("number of rows before cleaning:", df_food.count())
df_food = df_food.withColumn(
    "product_name",
    expr("filter(product_name, x -> lower(x.lang) = 'en')")
)

# Extract first product_name.text from array of structs
df_food = df_food.withColumn( "product_name",
    when(size("product_name") > 0, col("product_name").getItem(0)["text"]).otherwise(None))

# Define UDF to extract carbohydrates from list of Rows
@udf(DoubleType())
def extract_carbs_udf(nutriments_array):
    try:
        for item in nutriments_array:
            # This assumes item is a Row or dict-like object
            name = item.get("name", "").lower() if isinstance(item, dict) else getattr(item, "name", "").lower()
            if name == "carbohydrates":
                # Prefer "100g", fallback to "value"
                return item.get("100g") if isinstance(item, dict) else getattr(item, "100g", getattr(item, "value", None))
    except Exception as e:
        return None

df_food = df_food.withColumn("carbohydrates_100g", extract_carbs_udf(col("nutriments")))


df_food = df_food.select("product_name", "carbohydrates_100g")

# Drop rows with nulls or blank/empty strings
df_food = df_food.filter(
    (col("product_name").isNotNull()) &
    (trim(col("product_name")) != "") &
    (col("carbohydrates_100g").isNotNull())
)
print("number of rows after cleaning1:", df_food.count())
df_food = df_food.withColumn("product_name", df_food["product_name"].cast(StringType()))

# Function to remove accents
def remove_accents(input_str):
    if input_str is None:
        return None
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# Register UDF
remove_accents_udf = udf(remove_accents, StringType())

# lower case and remove accents

df_food = df_food.withColumn("product_name", lower(remove_accents_udf(col("product_name"))))


# Remove punctuation except hyphen and collapse spaces
df_food = df_food.withColumn(
    "product_name",
    trim(
        regexp_replace(
            regexp_replace("product_name", r"[^\w\s-]", ""),
            r"\s+", " "
        )
    )
)

# keep only rows where product_name contains at least two letters
df_food = df_food.filter(col("product_name").rlike(r"[a-zA-Z].*[a-zA-Z]"))


# Drop rows with nulls or blank/empty strings
df_food = df_food.filter(
    (col("product_name").isNotNull()) &
    (trim(col("product_name")) != "")
)
print("number of rows after cleaning2:", df_food.count())

p = inflect.engine()

def singularize_phrase(phrase):
    words = phrase.split()
    return ' '.join([p.singular_noun(w) if p.singular_noun(w) else w for w in words])

singularize_udf = udf(singularize_phrase, StringType())
df_food = df_food.withColumn("product_name", singularize_udf("product_name"))

df_food.show(20)

# Extract ingredient name using ingredient_parser
@pandas_udf(StringType())
def parse_ingredient_name_udf(product_names: pd.Series) -> pd.Series:
    results = []
    for sentence in product_names:
        try:
            parsed = parse_ingredient(sentence, separate_names=False)  # foundation_foods=True,
            name_texts = " ".join([n.text for n in parsed.name])
        except Exception as e:
            name_texts = sentence
        results.append(name_texts)
    return pd.Series(results)

df_food = df_food.withColumn("parsed_product_name", parse_ingredient_name_udf(df_food["product_name"]))

# # 2. Convert array to string (space-separated; use '' if you want no space)
# concat ws?????
# df_food = df_food.withColumn(
#     "parsed_product_name_str", concat_ws(" ", df_food["parsed_product_name"])
# )

df_food = df_food.drop("product_name")
df_food = df_food.withColumnRenamed("parsed_product_name", "product_name")

df_food.show(20)

# Drop rows with nulls or blank/empty strings
df_food = df_food.filter(
    (col("product_name").isNotNull()) &
    (trim(col("product_name")) != "")
)
print("number of rows after cleaning3:", df_food.count())

# Rename 'carbohydrates_100g' to 'carb_old'
df_food = df_food.withColumnRenamed("carbohydrates_100g", "carb_old")

# Create new 'carbohydrates_100g' column with median per 'product_name'
df_median = df_food.groupBy("product_name").agg(
    percentile_approx("carb_old", 0.5).alias("carbohydrates_100g")
)

df_food = df_food.drop("carb_old").join(broadcast(df_median), on="product_name", how="left")

df_food = df_food.withColumn("carbohydrates_100g", col("carbohydrates_100g").cast(FloatType()))


df_food = df_food.dropDuplicates(["product_name"])


print("number of rows in df_food:",df_food.count())

# Save the cleaned dataframe in the same directory as the original food.parquet
df_food.write.mode("overwrite").parquet(OUT_PATH)
print(f"Saved cleaned df_food to: {OUT_PATH}")

print("Current working directory:", os.getcwd())
PATH = os.path.abspath(os.path.join(os.getcwd(), 'df_food_parsed.parquet'))

new_df = spark.read.parquet(PATH)
print("number of rows in new df_food:",new_df.count())

spark.stop()