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
import geoviews as gv
import geoviews.tile_sources as gvts
from geoviews import dim, opts
import geopandas as gpd
import hvplot.pandas
gv.extension('bokeh', 'matplotlib')
import geoviews.feature as gf

os.environ['JAVA_HOME']="/Library/Java/JavaVirtualMachines/jdk1.8.0_251.jdk/Contents/Home"
'''
About the dataset after exploratory analysis:
Total Count of Records - 346544
Records with non null Population information - 300333
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
    #### Lets join with the metadata to get the latitude and longitude and population
    merged = spark.sql("""select d.regionId,
    referenceDate,
    totalDeaths,
    totalConfirmedCases,
    totalRecoveredCases,
    totalTestedCases,
    numPositiveTests,
    numDeaths,
    numRecoveredCases,
    avgWeeklyDeaths,
    avgWeeklyConfirmedCases,
    avgWeeklyRecoveredCases,
    m.woeId,m.wikiId,m.longitude,m.latitude,m.population
    from data d join metadata m on
    d.regionId=m.id 
    and m.population is not null""").toPandas()#.show(truncate=False)
    print(merged.head())
    null_columns = merged.columns[merged.isnull().any()]
    print(merged[null_columns].isnull().sum())
    #gv.Points(merged['latitude'],merged['longitude'])
    # kdims = ['longitude', 'latitude']
    # vdims = ['population']
    # population_map = gv.Dataset(merged, kdims=kdims,vdims=vdims)
    # gv.Image(population_map)
    merged['DateTime'] = merged['referenceDate'].apply(lambda x: pd.to_datetime(str(x), format='%Y-%m-%d'))
    img = merged.hvplot.points('longitude', 'latitude', geo=True, color='yellow',
                               alpha=0.2, xlim=(-180, -30), ylim=(0, 72), tiles='ESRI')
    hvplot.save(img,'covidmap.html')
    tested_dead_recovered = merged.hvplot.bar(x='totalTestedCases', y=['totalDeaths', 'totalRecoveredCases'],
                     stacked=True, rot=90, width=800, legend='top_left')
    hvplot.save(tested_dead_recovered,'tested_vs_dead.html')

if __name__ == "__main__":
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    print("Exploring datasets")
    explore_data_by_time(spark)
    spark.stop()



########
# fig = plt.figure(figsize=(10, 5))
#     ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
#     # # make the map global rather than have it zoom in to
#     # # the extents of any plotted data
#     ax.set_global()
#     ax.stock_img()
#     ax.coastlines()
#     ax.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
#     ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree())
#     ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic())
#     plt.show()