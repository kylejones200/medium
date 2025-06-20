from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

spark = SparkSession.builder.appName("MLPipeline").getOrCreate()

df = spark.read.csv("data.csv", header=True, inferSchema=True)
vec = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
df = vec.transform(df).select("features", "label")

lr = LogisticRegression()
model = lr.fit(df)
predictions = model.transform(df)
predictions.show()