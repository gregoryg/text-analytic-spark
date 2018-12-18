from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext


from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import udf, expr, concat, col
import struct
from pyspark.sql.functions import monotonically_increasing_id, lit, regexp_replace
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
import re

conf = SparkConf().setAppName("text-analytics-flight")
sc = SparkContext(conf=conf)
sql = SQLContext(sc)

# # ML Analysis of free form text
# This code leverages the power of the Cloudera Spark engine to analyze free-form text - 
# reviews of airline experiences, in this case.

# # Initial data load and data cleansing
# Let's start by reading the data in, giving the rows unique IDs
# Additionally, we will cast numeric columns and get rid of quotation marks
rawdata = sql.read.load("hdfs:///user/gregj/data/airline-reviews/airlines.csv", format="csv", header=True)
rawdata = rawdata.fillna({'review': ''})                               # Replace nulls with blank string
rawdata = rawdata.withColumn("uid", monotonically_increasing_id())     # Create Unique ID
rawdata = rawdata.withColumn("year_month", rawdata.date.substr(1,7))   # Generate YYYY-MM variable
rawdata = rawdata.withColumn("rating", rawdata.rating.cast(IntegerType()))
rawdata = rawdata.withColumn("value", rawdata.value.cast(IntegerType()))
rawdata = rawdata.withColumn("review", regexp_replace('review', '"', ''))

# Show rawdata (as DataFrame)
rawdata.show(10
             
# Our next step is to remove "stop words" - commonly occuring connecting words, and to vectorize the resulting list of words, preparing the text for submission to ML analysis

def cleanup_text(mytext):
    words = mytext.split()
    
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = ['']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out

udf_cleantext = udf(cleanup_text , ArrayType(StringType()))
rawdata = rawdata.withColumn("review_filtered", udf_cleantext(rawdata.review))

# Machine Learning on Text

# And now the good stuff begins.  We start by running a CountVectorizer from the SparkML Features library
# This wil count word frequencies in preparation for indexing by TF-IDF and other methods
             
# cv = CountVectorizer(inputCol="review_filtered", outputCol="tf", minTF=1, vocabSize=1000)
cv = CountVectorizer(inputCol="review_filtered", outputCol="rawFeatures", vocabSize=1000)
cvmodel = cv.fit(rawdata)
featurizedData = cvmodel.transform(rawdata)

# We now have the complete vocabulary, with frequency per review
# We will "broadcast" that vocabulary acroos all nodes in the cluster, to permit relative frequency indexing across reviews
vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)

# # TF-IDF (Term Frequency-Inverse Document Frequency)
# In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.[1] It is often used as a weighting factor in searches of information retrieval, text mining, and user modeling. T
             
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData) # TFIDF

## Use LDA to cluster the tf-idf matrix
# Generate 25 Data-Driven Topics:
# "em" = expectation-maximization 
lda = LDA(k=25, seed=123, optimizer="em", featuresCol="features")
ldamodel = lda.fit(rescaledData)
 
ldatopics = ldamodel.describeTopics()
## Show the top 25 Topics
ldatopics.show(25)

def map_termID_to_Word(termIndices):
    words = []
    for termID in termIndices:
        words.append(vocab_broadcast.value[termID])
    
    return words

udf_map_termID_to_Word = udf(map_termID_to_Word , ArrayType(StringType()))
ldatopics_mapped = ldatopics.withColumn("topic_desc", udf_map_termID_to_Word(ldatopics.termIndices))
ldatopics_mapped.select(ldatopics_mapped.topic, ldatopics_mapped.topic_desc).show(50,False)


# # Combine the data-driven topics with the original airlines dataset
# Now, let's join the original data in with the computed statistics to make the new insights accessible to a broad range of analysts.
ldaResults = ldamodel.transform(rescaledData)
ldaResults.select('id','airline','date','cabin','rating','review_filtered','features','topicDistribution').show()

def breakout_array(index_number, record):
    vectorlist = record.tolist()
    return vectorlist[index_number]

udf_breakout_array = udf(breakout_array, FloatType())

# ## Extract document weights for Topics 12 and 20
# Focus in on details for a couple of topics of particular interest to our analysts
enrichedData = ldaResults                                                                   \
        .withColumn("Topic_12", udf_breakout_array(lit(12), ldaResults.topicDistribution))  \
        .withColumn("topic_20", udf_breakout_array(lit(20), ldaResults.topicDistribution))            

enrichedData.select('id','airline','date','cabin','rating','review_filtered','features','topicDistribution','Topic_12','Topic_20').show()

enrichedData.createOrReplaceTempView("enrichedData")

# # Persist the ML data along to an SQL table
sql.sql('create table if not exists airlines.enriched_reviews STORED AS parquet AS select * FROM enrichedData')
