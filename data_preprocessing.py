from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, collect_list, sort_array, udf
from pyspark.sql.types import StringType
import pandas as pd  # For small-scale testing
from sklearn.model_selection import train_test_split

# Initialize 
spark = SparkSession.builder \
    .appName("ProductRecommendationPreprocessing") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

data = {
    'user_id': ['user1', 'user1', 'user2', 'user2', 'user3'],
    'product_id': ['prodA', 'prodB', 'prodA', 'prodC', 'prodD'],
    'review_text': ['Great product!', 'Okay, but could be better.', 'Loved it!', 'Not recommended.', 'Excellent!'],
    'rating': [5, 3, 5, 2, 4],
    'timestamp': ['2025-01-01', '2025-02-01', '2025-01-15', '2025-03-01', '2025-04-01']
}
pdf = pd.DataFrame(data)
df = spark.createDataFrame(pdf)


df = df.orderBy('user_id', 'timestamp')
grouped = df.groupBy('user_id').agg(
    sort_array(collect_list(concat_ws(':', col('product_id'), col('review_text')))).alias('sequence')
)


def format_sequence(seq):
    return "User history: " + " ; ".join(seq) + " ; Recommended next: [MASK]"
format_udf = udf(format_sequence, StringType())
grouped = grouped.withColumn('input_text', format_udf(col('sequence')))

def split_sequence(seq):
    if len(seq) > 1:
        history = " ; ".join(seq[:-1])
        target = seq[-1].split(':')[0]  # Product ID
        return f"User history: {history} ; Recommended next:", target
    return None, None
split_udf = udf(split_sequence, StringType())  
grouped.write.parquet("preprocessed_data.parquet", mode="overwrite")

# Split into train/test 
pandas_df = grouped.toPandas()
train_df, test_df = train_test_split(pandas_df, test_size=0.2)
train_df.to_parquet("train.parquet")
test_df.to_parquet("test.parquet")

spark.stop()
print("Data preprocessing complete.")