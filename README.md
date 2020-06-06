Dataset from  https://github.com/yahoo/covid-19-data.git
The folder yahoo-data contains the tsv files from https://github.com/yahoo/covid-19-data/tree/master/data

For the maps to work, we need to follow the installation instructions for Matplotlib toolkit
as below: (https://matplotlib.org/basemap/users/installing.html)
Activate your corresponding virtual environment:
In this case `source /Users/sunitakoppar/PycharmProjects/covid_analysis/venv/bin/activate`


In  case of challenges like I had with step 4, try the below troubleshooting steps:
1. brew update && brew upgrade
2. brew uninstall --ignore-dependencies openssl
3. brew install openssl
`References:`
https://medium.com/civis-analytics/prediction-at-scale-with-scikit-learn-and-pyspark-pandas-udfs-51d5ebfb2cd8

https://makersportal.com/blog/2018/7/20/geographic-mapping-from-a-csv-file-using-python-and-basemap




------------------------------------------------------------------------------------
Attempts to use basemap which is older version of Cartopy.
1. brew install proj
2. brew install geos
3. Download Opensource Geometry Engine from here http://download.osgeo.org/geos/geos-3.8.1.tar.bz2.
    bunzip2 geos-3.8.1.tar.bz2;  cd geo-3.8.1; export GEOS_DIR=/usr/local/share; ./configure --prefix=$GEOS_DIR; make; make install
4. Download https://github.com/matplotlib/basemap/archive/v1.1.0.tar.gz
 gunzip basemap-1.1.0.tar.gz; tar -xvf basemap-1.1.0.tar; cd basemap-1.1.0; sudo su; export GEOS_DIR=/usr/local/share; python setup.py install
