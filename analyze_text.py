# # Load the data
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import udf, expr, concat, col
import struct
from pyspark.sql.functions import monotonically_increasing_id
import pandas as pd
from pyspark.ml.feature import CountVectorizer
from sklearn.model_selection import train_test_split
import pyspark.ml.feature as feature
from pyspark.ml import Pipeline


conf = SparkConf().setAppName("text-analytics-flight")
sc = SparkContext(conf=conf)
sql = SQLContext(sc)

rawdata = sql.read.load("hdfs:///user/gregj/data/airline-reviews/airlines.csv", format="csv", header=True)
rawdata = rawdata.fillna({'review': ''})                               # Replace nulls with blank string
rawdata = rawdata.withColumn("uid", monotonically_increasing_id())     # Create Unique ID
rawdata = rawdata.withColumn("year_month", rawdata.date.substr(1,7))   # Generate YYYY-MM variable
 
# Show rawdata (as DataFrame)
rawdata.show(10)

# Print data types
for type in rawdata.dtypes:
    print (type)

target = rawdata.select(rawdata['rating'].cast(IntegerType()))
target.dtypes

## GJG - filter stopwords, index words
tokenizer = feature.Tokenizer(inputCol="review", outputCol="review_array")
sw = feature.StopWordsRemover(inputCol="review_array", outputCol="review_filtered")
# cv = feature.CountVectorizer(inputCol="review_filtered", outputCol="tf", minTF=1, vocabSize=2 ** 17)
cv = feature.CountVectorizer(inputCol="review_filtered", outputCol="tf", minTF=1, vocabSize=1000)
cv_transformer = Pipeline(stages=[tokenizer, sw, cv]).fit(rawdata)
idf = feature.IDF(minDocFreq=10, inputCol="tf", outputCol="tfidf")
tfidf_transformer = Pipeline(stages=[cv_transformer, idf]).fit(rawdata)
rawdata = tfidf_transformer.transform(rawdata)

rawdata.registerTempTable('reviews')


# train, test = train_test_split(rawdatapd, test_size=0.2)

################################################################################################
#
#   Generate TFIDF
#
################################################################################################

# Term Frequency Vectorization  - Option 1 (Using hashingTF): 
#hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
#featurizedData = hashingTF.transform(clean_text)

# Term Frequency Vectorization  - Option 2 (CountVectorizer)    : 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 1000)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)
