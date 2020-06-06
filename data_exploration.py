import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
import os
import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

os.environ['JAVA_HOME']="/Library/Java/JavaVirtualMachines/jdk1.8.0_251.jdk/Contents/Home"
'''
About the dataset after exploratory analysis:
Count of Records - 346544
Distinct Regions/label - 6092
Distinct DataSources - 226

With following structure for the datasets:
 |-- regionId: string (nullable = true)
 |-- label: string (nullable = true)
 |-- referenceDate: string (nullable = true)
 |-- lastUpdatedDate: string (nullable = true)
 |-- totalDeaths: string (nullable = true)
 |-- totalConfirmedCases: string (nullable = true)
 |-- totalRecoveredCases: string (nullable = true)
 |-- totalTestedCases: string (nullable = true)
 |-- numPositiveTests: string (nullable = true)
 |-- numDeaths: string (nullable = true)
 |-- numRecoveredCases: string (nullable = true)
 |-- diffNumPositiveTests: string (nullable = true)
 |-- diffNumDeaths: string (nullable = true)
 |-- avgWeeklyDeaths: string (nullable = true)
 |-- avgWeeklyConfirmedCases: string (nullable = true)
 |-- avgWeeklyRecoveredCases: string (nullable = true)
 |-- dataSource: string (nullable = true)
 
 And below structure for the metadata:
 root
 |-- id: string (nullable = true)
 |-- type: string (nullable = true)
 |-- woeId: string (nullable = true)
 |-- label: string (nullable = true)
 |-- wikiId: string (nullable = true)
 |-- longitude: string (nullable = true)
 |-- latitude: string (nullable = true)
 |-- population: string (nullable = true)
 |-- parentId: string (nullable = true)

'''

def explore_data_by_time(spark):
    df=spark.read.format("csv")\
        .option("header", "true")\
        .option("delimiter", "\t")\
        .load("./yahoo-data/*.tsv")
    df.createOrReplaceTempView("data")
    metadata = spark.read.format("csv") \
        .option("header", "true") \
        .option("delimiter", "\t") \
        .load("./yahoo-data/metadata/*.tsv")
    metadata.createOrReplaceTempView("metadata")
    #### Below are some exploratory queries I will comment after initial exploration:
    # spark.sql("""select label , sum(totalDeaths),sum(totalConfirmedCases),sum(totalRecoveredCases)
    #            from data group by label, regionId""").show()
    # distinct_regions = spark.sql("""select distinct dataSource from data""").show(truncate=False)
    # print(distinct_regions.count())
    #### Lets join with the metadata to get the latitude and longitude
    metadata.printSchema()
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic())
    plt.show()


if __name__ == "__main__":
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    print("Exploring datasets")
    explore_data_by_time(spark)
    spark.stop()