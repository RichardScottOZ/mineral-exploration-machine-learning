# mineral-exploration-machine-learning
This page lists resources for mineral exploration and machine learning, generally with useful code and examples. 
ML and Data Science is a huge field, these are resources I have found useful and/or interesting to me in practice.
Links currently to a fork of a repository are because I have changed something to use and put in a list for reference.
Resources are also given for data analysis, transformation and visualisation as that is most of the work.

# Table of Contents

* [Prospectivity](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#prospectivity)
* [Geology](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#geology)
* [Natural Language Processing](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#natural-language-processing)
* [Remote Sensing](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#natural-language-processing)
* [Data Quality](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#data-quality)
* [Community](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#communit)
* [Cloud providers](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#cloud-providers)
* [Domains](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#domains)
* [Overview](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#overview)
* [Web Services](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#web-services)
* [Data Portals](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#data-portals)
* [Tools](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#tools)
* [Ontologies](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#ontologies)
* [Books](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#books)
* [Papers](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#papers)
* [Other](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#other)

# Frameworks
* [UNCOVER-ML Framework](https://github.com/RichardScottOZ/uncover-ml)
  * [Geo-Wavelets](https://github.com/RichardScottOZ/geo-wavelets)
  * [ML-Preprocessing](https://github.com/GeoscienceAustralia/ML-preprocessing)
  * [GIS ML Workflow](https://github.com/sheecegardezi/GIS-ML-Workflow)
* [PySpatialML](https://github.com/RichardScottOZ/Pyspatialml) -> Library that facilitates prediction and handling for raster machine learning automatically to geotiff, etc.
*[TorchGeo](https://github.com/microsoft/torchgeo) -> Pytorch library for remote sensing style models
* [Geo Deep Learning](https://github.com/RichardScottOZ/geo-deep-learning) -> Simple deep learning framework based on RGB


# Mineral Prospectivity
* [Transform 2022 Tutorial](https://github.com/Solve-Geosolutions/transform_2022) -> Random forest example
  * [Video](https://www.youtube.com/watch?v=C4YvnLMzYDc)
## Explorer Challenge
* [Explorer Challenge](https://github.com/RichardScottOZ/explore_australia) -> OZ Minerals run competition with Data Science introduction
## Explore SA - South Australian Department of Energy and Mining Competition
* [Caldera](https://github.com/mrodda117/CalderaPublic) -> Caldera Analytics analysis
* [IncertoData](https://github.com/RichardScottOZ/ExploreSA/tree/master/Data_submission_competition)
* [Butterworth and Barnett](https://github.com/RichardScottOZ/gawler-exploration) -> Butterworth and Barnett entry
## Brazil
* [Mapa Preditivo](https://github.com/fnaghetini/Mapa-Preditivo) -> Brazil student project
* [Mineral Prospectivity Mapping](https://github.com/Eliasmgprado/MineralProspectivityMapping)
* [3D Weights of Evidence](https://github.com/e-farahbakhsh/3DWofE)
## Commodities
* [Tin-Tungsten](https://medium.com/@thomas.ostersen/tin-tungsten-prospecting-with-machine-learning-in-northeast-tasmania-australia-3c23519f81cf)
  * [Collab](https://colab.research.google.com/drive/168PSo21-Jkwdz8xOmr5-rX9_DL3SInCN?usp=sharing)
* [Pb-Zn-Ag-Au](https://github.com/ZhiqiangZhangCUGB/MPM-by-ensemble-learning) -> Qingchengzi Pb-Zn-Ag-Au polymetallic district China
# Geology
* [Brazil Predictive Geology Maps](https://github.com/marcosbr/predictive-geology-maps) -> Work by the Brazil geological survey based on their datasets
* [mapeamento_litologico_preditivo](https://github.com/Gabriel-Goes/mapeamento_litologico_preditivo)
* [Neural Rock Typing](https://github.com/LukasMosser/neural_rock_typing)
* [West Musgraves Geology Uncertainty](https://medium.com/@thomas.ostersen/uncertainty-mapping-in-the-west-musgraves-australia-988fc49ce1e4) -> Uncertainty map prediction with entropy analysis: highly useful
  * [Collab](https://colab.research.google.com/drive/1pPZCSjlNPn7_n8GLGCxY0bo7QmbVU65G?usp=sharing) -> Notebook
## Lithology
* [Deep Learning Lithology](https://github.com/RichardScottOZ/deeplearning_lithology)
* [Rock Protolith Predictor](https://github.com/RichardScottOZ/Rock_protolith_predictor)
* [SA Geology Lithology Predictions](https://github.com/RADutchie/SA-geology-litho-predictions)
## Structure
* [Lineament Learning](https://github.com/aminrd/LineamentLearning) -> Fault prediction and mapping via potential field deep learning and clustering

# Geochemistry
* [Dash Geochemical Prospection](https://github.com/pvabreu7/DashGeochemicalProspection) -> Web-app classifying stream sediments with K-means
* [ICBMS Jacobina](https://github.com/gferrsilva/icpms-jacobina) -> Analysis of pyrite chemistry from a gold deposit
* [LewisML](https://github.com/RichardScottOZ/LewisML) -> Analysis of the Lewis Formation
* [Global geochemistry](https://github.com/dhasterok/global_geochemistry)
* [QMineral Modeller](https://github.com/gferrsilva/QMineral_Modeller) -> Mineral Chemistry virtual assistant from the Brazilian geological survey

# Natural Language Processing
* [Text Extraction](https://github.com/RichardScottOZ/amazon-textract-textractor) -> Text extraction from documents : paid ML as a service, but works very well, can extract tables efficiently
  * [Large Scale](https://github.com/RichardScottOZ/amazon-textract-serverless-large-scale-document-processing) -> Large scale version
* [NASA Concept Tagging](https://github.com/RichardScottOZ/concept-tagging-training) -> Keyword prediction api
* [Petrography Report Data Extractor](https://github.com/RichardScottOZ/Petrography-report-data-extractor)
* [SA Exploration Topic Modelling](https://github.com/RADutchie/SA-exploration-topic-modelling) -> Topic modelling from exploration reports
* [Geo NER Model](https://github.com/BritishGeologicalSurvey/geo-ner-model) -> Named entity recognition
* [Stratigraph](https://github.com/BritishGeologicalSurvey/stratigraph)
* [Geocorpus](https://github.com/jneto04/geocorpus)
* [Portuguese BERT](https://github.com/neuralmind-ai/portuguese-bert)
## Word Embedings
* [Geoscience Language Models](https://github.com/NRCan/geoscience_language_models) -> processing code pipeline and models [Glove, BERT) retrained on geoscience documents from Canada
* [GeoVec](https://github.com/spadarian/GeoVec) -> Word embedding model trained on 300K geoscience papers
  * [GeoVec Model](https://osf.io/4uyeq/) -> OSF Storage for GeoVec model
    * [Paper](https://soil.copernicus.org/articles/5/177/2019/)
    * [GeoVecto Litho](https://github.com/IFuentesSR/GeoVectoLitho) -> 3D Models interpolation from word embeddings
  * [GeoVEC Playground](https://github.com/RichardScottOZ/geoVec-playground) -> Working with the Padarian GeoVec glove word embeddings model
* [GloVe](https://github.com/stanfordnlp/GloVe) -> Standford library for producing word embeddings
  * [gloVE python](https://pypi.org/project/glove-python-binary/) glove, glove-python highly problematic on windows: here Binary version for Windows installs:
* [Mittens](https://github.com/roamanalytics/mittens) -> In memory vectorized glove implementation 
* [PetroVec](https://github.com/Petroles/Petrovec) -> Portuguese Word Embeddings for the Oil and Gas Industry: development and evaluation
* [wordembeddingsOG](https://github.com/diogosmg/wordEmbeddingsOG) -> Portuguese Oil and Gas word embeddings
* [Portuguese Word Embeddings(https://github.com/nathanshartmann/portuguese_word_embeddings) 
* [Spanish Word Embeddings](https://github.com/dccuchile/spanish-word-embeddings)

# Remote Sensing
* [CNN Sentinel](https://github.com/jensleitloff/CNN-Sentinel) -> Overview about land-use classification from satellite data with CNNs based on an open dataset
## Spectral Unmixing
* [Hyperspectral Autoencoders](https://github.com/RichardScottOZ/hyperspectral-autoencoders)
* [Deeplearn HSI](https://github.com/hantek/deeplearn_hsi)
* [3DCAE-hyperspectral-classification](https://github.com/MeiShaohui/3DCAE-hyperspectral-classification)
* [DeHIC](https://github.com/jingge326/DeHIC)
* [Pysptools](https://github.com/RichardScottOZ/pysptools) -> also has useful heuristic algorithms
* [Spectral Python](https://github.com/spectralpython/spectral)
* [Spectral Dataset RockSL](https://github.com/RichardScottOZ/spectral-dataset-RockSL) -> Open spectral dataset
* [Unmixing](https://github.com/RichardScottOZ/unmixing)

## Course
[ML Geoscience](https://github.com/DIG-Kaust/MLgeoscience)
[Geoanalytics and machine learning](https://github.com/victsnet/Geoanalytics-and-Machine-Learning)


## Platforms
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) -> Computing platform connected to data sources
## Guides
* [Satellite Image Deep Learning](https://github.com/robmarkcole/satellite-image-deep-learning)
* [Earth Observation](https://github.com/RichardScottOZ/awesome-earthobservation-code)


# Data Quality
* [Data Quality](https://github.com/RichardScottOZ/Geoscience-Data-Quality-for-Machine-Learning)
* [Australian Gravity Data](https://github.com/RichardScottOZ/australia-gravity-data) -> Overview and analysis of gravity station data
* [Geodiff[(https://github.com/MerginMaps/geodiff) -> Comparison of vector datasets
* [Redflag](https://github.com/agilescientific/redflag) -> Analysis of datasets and overview to detect problems
  
# Machine Learning
[Geospatial-ml](https://github.com/giswqs/geospatial-ml) -> Install multiple common packages at once
[Dask-ml](https://github.com/dask/dask-ml) -> Distributed versions of some common ML algorithms
## Probabilistic
  * [NG Boost](https://github.com/stanfordmlgroup/ngboost) -> probabilistic regression
  * [Probabilistic ML](https://github.com/ZhiqiangZhangCUGB/Probabilistic-machine-learning)
    * [Bagging PU with BO](https://github.com/ZhiqiangZhangCUGB/Bagging-PU-with-BO)
## Clustering  
### Self Organising Maps
  * [GisSOM](https://github.com/RichardScottOZ/GisSOM) -> Geospatial centred Self Organising Maps from Finland Geological Survey
  * [SimpSOM](https://github.com/fcomitani/SimpSOM) -> Self Organising Maps 
* [hdbscan](https://github.com/scikit-learn-contrib/hdbscan)
* [kmedoids](https://github.com/letiantian/kmedoids)
* [Picasso](https://github.com/pachterlab/picasso)
## Explainability
* [InterpretML](https://github.com/interpretml/interpret-community) -> Interpreting models of tabular data

## Deep Learning
* [Deep Colormap Extraction](https://github.com/RichardScottOZ/deep_colormap_extraction) -> Trying to extract a data scale from pictures
* [Extract and Classify Images from Geoscience Documents](https://github.com/bolgebrygg/extract-and-classify-images-from-geoscience-documents)
### Data
* [Xbatcher](https://github.com/RichardScottOZ/xbatcher) -> Xarray based data reading for deep learning
### Explainability
* [Shap Values](https://github.com/slundberg/shap)

### Semi-supervised learning
* [Self Supervised](https://github.com/untitled-ai/self_supervised) -> Pytorch lightning implementations of multiple algorithms
* [Simclr](https://github.com/google-research/simclr)
## Hyperparameters
* [Hyperopt](https://github.com/hyperopt/hyperopt)
* [TPOT Automated ML](https://github.com/trhallam/tpot)
  
## Coding Environments
* [DEA Sandbox](https://docs.dea.ga.gov.au/setup/Sandbox/sandbox.html)
* [Cube In A Box](https://github.com/RichardScottOZ/cube-in-a-box)

# Community
* [Software Underground](https://softwareunderground.org/) - Community of people interested in exploring the intersection of the subsurface and code
  * [Geoscience Open Source Tie-In](https://github.com/RichardScottOZ/gostin)
  * [Slack Channel](https://softwareunderground.org/slack)
  * [Videos](https://www.youtube.com/c/SoftwareUnderground/videos)
    * [Transform 2022](https://www.youtube.com/playlist?list=PLgLft9vxdduDFkG9gtuNicNmb2YUzWqSQ)
  * [Awesome Open Geoscience](https://github.com/softwareunderground/awesome-open-geoscience )[note Oil and Gas Biased]
  * [Transform 2021 Hacking Examples](https://github.com/RichardScottOZ/Transform-2021)
  * [Segysak 2021 Tutorial](https://github.com/trhallam/segysak-t21-tutorial)
* [Pangeo](https://pangeo.io/)
  * [Forum]
  * [COG Best Practices](https://github.com/pangeo-data/cog-best-practices)
* [Digital Earth Australia](https://www.dea.ga.gov.au/)
  * [Slack Channel]
* [Open Source Geospatial Foundation](https://github.com/OSGeo/osgeo)
* [ASEG](https://www.youtube.com/c/ASEGVideos/videos) -> videos from Australia Society of Exploration Geoscientists

# Cloud Providers
## AWS
* [ec2 Spot Labs](https://github.com/awslabs/ec2-spot-labs) -> Making automatically working sith Spot instances easier
* [Mlmax](https://github.com/awslabs/mlmax) - Start fast library
* [Sagemaker](https://github.com/aws/amazon-sagemaker-examples) -> ML Managed Service
  * [SDK](https://github.com/aws/sagemaker-python-sdk)
  * [Entrypoint Utilities](https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities)
  * [Workshop 101](https://github.com/RichardScottOZ/sagemaker-workshop-101)
  * [Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)
* [Shepard](https://github.com/RichardScottOZ/shepard) -> Automated cloud formation setup of AWS Batch Pipelines: this is great
* [Smallmatter](https://github.com/aws-samples/smallmatter-package)
* [Pyutil](https://github.com/verdimrc/pyutil)
* [Deep Learning Containers](https://github.com/aws/deep-learning-containers)
* [Loguru](https://github.com/Delgan/loguru) -> Logging library
* [AWS GDAL Robot](https://github.com/mblackgeo/aws-gdal-robot) -> Lambda and batch processing of geotiffs
* [Serverless Seismic Processing](https://github.com/vavourak/serverless_seismic_processing)

# Overviews
* [Mineral Exploration](https://www.ga.gov.au/scientific-topics/minerals/mineral-exploration)
## Domains
* [Geology](https://en.wikipedia.org/wiki/Geology)
  * [Geologic Ages](https://en.wikipedia.org/wiki/Geologic_time_scale)
  * [Lithology](https://en.wikipedia.org/wiki/Lithology)
  * [Stratigraphy]() 
* [Geochemistry](https://en.wikipedia.org/wiki/Geochemistry)
* [Geophysics](https://en.wikipedia.org/wiki/Geophysics)
* [Remote Sensing]()

# Web Services
* [AusGIN](https://www.geoscience.gov.au/web-services)
* [Geoscience Australia](http://services.ga.gov.au/)
* [NSW](https://www.regional.nsw.gov.au/meg/geoscience/products-and-data/gis-web-services)
* [Queensland](https://gisservices.information.qld.gov.au/arcgis/rest/services)
* [SARIG](https://map.sarig.sa.gov.au/MapViewer/StartUp/?siteParams=WebServicesWidget)
* [Tasmania WFS](https://www.mrt.tas.gov.au/products/digital_data/web_feature_service)
* [Victoria Geonetwork](http://geology.data.vic.gov.au/)

[PyESRIDump](https://github.com/RichardScottOZ/pyesridump) -> Library to grab data at scale from ESRI Rest Servers

# APIs
* [Open Data API](https://github.com/RichardScottOZ/open-data-api) -> GSQ Open Data Portal API
  * [Geochemistry parsing](https://github.com/geological-survey-of-queensland/geochemistry_parsing)

# Data Portals
* [SARIG](https://map.sarig.sa.gov.au/)
  * [Reports](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/cat0/MSearch;jsessionid=492C6538B64080CE8B13E91C79F8B1BA)
  * [Seismic](https://www.petroleum.sa.gov.au/data-centre/seismic-data)
* [Geoscience Australia Portal](https://portal.ga.gov.au/)
* [Exploring for the Future Portal](https://portal.ga.gov.au//eftf)
* [GSQ](https://geoscience.data.qld.gov.au/)
* [GEOVIEW](https://geoview.dmp.wa.gov.au/geoview/?Viewer=GeoView)
* [STRIKE](https://strike.nt.gov.au/wss.html)
* [MINVIEW](https://minview.geoscience.nsw.gov.au/)
* [Natural Resources Canada](https://www.nrcan.gc.ca/earth-sciences/geography/atlas-canada/explore-our-data/16892
## Reports
* [NT](https://geoscience.nt.gov.au/gemis)
* [SARIG](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/cat0/MSearch)
* [WAMEX](https://www.dmp.wa.gov.au/Geological-Survey/GSWA-publications-and-maps-1399.aspx)

# Tools
## GIS
  * [QGIS](https://qgis.org/en/site/) -> GIS Data Visualisation and Analysis Open Source desktop application, has some ML tools
  * [GRASS](https://github.com/OSGeo/grass)
## 3D
* [PyVista](https://github.com/pyvista/pyvista) -> VTK wrapping api for great data visualisation and analysis
  * [PVGeo](https://pvgeo.org/index.html)
  * [Pyvista-Xarray](https://github.com/RichardScottOZ/pyvista-xarray) -> Transforming xarray data to VTK 3D painlessly
  * [OMFVista[(https://github.com/OpenGeoVis/omfvista0 ->Pyvista for Open Mining Format
  * [Scipy 2022 Tutorial](https://github.com/pyvista/pyvista-tutorial)
* [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) -> Mesh transformation
* [Open Mining Format](https://github.com/gmggroup/omf)
* [Whitebox Tools](https://github.com/jblindsay/whitebox-tools)
  * [GUI](https://github.com/giswqs/whiteboxgui) -> Desktop version
* [Subsurface](https://github.com/softwareunderground/subsurface)
* [Geolambda](https://github.com/bluetyson/geolambda) -> AWS Lambda setup
* [Geoscience Analyst](https://mirageoscience.com/mining-industry-software/geoscience-analyst/)
  * [geoh5py](https://geoh5py.readthedocs.io/) -> getting data to and from geoh5 projects
  * [geoapps](https://geoapps.readthedocs.io/en/stable/) -> notebook based applications for geophysics via geoh5py
  * [gams](https://github.com/eroots/gams) -> magnetic data analysis
* [Rayshader](https://github.com/tylermorganwall/rayshader)
    
## Geospatial General
* [Python resources for earth science](https://github.com/javedali99/python-resources-for-earth-sciences)
## Vector Data
* [Geopandas](https://geopandas.org/en/stable/)
  * [Dask-geopandas](https://github.com/RichardScottOZ/dask-geopandas)
    * [Tutorial](https://github.com/martinfleis/dask-geopandas-tutorial)
* [SF](https://r-spatial.github.io/sf/)

## Raster Data
* [Rasterio](https://github.com/rasterio/rasterio) -> python base library for raster data handling
* [Xarray](https://github.com/pydata/xarray) -> Multidimensional Labelled array handling and analysis
  * [Rioxarray](https://corteva.github.io/rioxarray/stable/) -> Fabulous high level api for xarray handling of raster data
  * [ODC-GEO](https://github.com/opendatacube/odc-geo/) -> Tools for remote sensing based raster handling with many extremely tools like colorisation, grid workflows
  * [Geocube](https://github.com/corteva/geocube) -> Rasterisation of vector data api
  * [COG Validator](https://github.com/rouault/cog_validator) -> checking format of cloud optimised geotiffs
  * [Xarray Spatial](https://github.com/RichardScottOZ/xarray-spatial) -> Statistical analysis of raster data such as classification like natural breaks
  * [xrft](https://github.com/RichardScottOZ/xrft) -> Xarray based Fourier Transforms
* [Raster](https://rspatial.org/raster/spatial/8-rastermanip.html) -> R library
    
## Data Conversion
* [CGG Outfile reader](https://github.com/RichardScottOZ/CGG-Out-Reader)
* [Geosoft Grid to Raster](https://github.com/RichardScottOZ/Geosoft-Grid-to-Raster)
* [GOCAD SG Grid Reader](https://github.com/RichardScottOZ/GOCAD_SG_Grid_Reader)
	* [geomodel-2-3dweb](https://github.com/RichardScottOZ/geomodel-2-3dweb) >- In here they have a method to extract data from binary GOCAD SG Grids
* [Leaprfrog Mesh Reader](https://github.com/ThomasMGeo/leapfrogmshreader)
* [VTK to DXF](https://github.com/RichardScottOZ/VTK-to-DXF)
## Geochemistry
* [Pygeochemtools](https://github.com/RADutchie/pygeochemtools) -> library and command line to enable rapid QC and plotting of geochemical data
* [SA Geochemical Maps](https://github.com/GeologicalSurveySouthAustralia/SA-geochemical-maps) -> Data Analysis and plotting of South Australia geochemistry data from the Geological Survey of SA
* [Geochemical levenning](https://github.com/GeoscienceAustralia/geochemical-levelling)
* [Periodic Table](https://github.com/pkienzle/periodictable)
## Geochronology
* [Geologic Time Scale](https://github.com/RichardScottOZ/GeologicTimeScale) -> Code to produce, but also has a nice regular csv of the Ages
## Geology
* [Gempy](https://github.com/RichardScottOZ/gempy) -> Implicit Modelling
* [Gemgis](https://github.com/cgre-aachen/gemgis) -> Geospatial Data Analysis assistance 
* [LoopStructural](https://github.com/Loop3D/LoopStructural) -> Implicity Modelling
* [Manual python geologia](https://github.com/kevinalexandr19/manual-python-geologia) -> Analysis of geology data
* [Map2Loop](https://github.com/Loop3D/map2loop-2) -> 3D Modelling Automation
  * [Loop3D](https://github.com/Loop3D/Loop3D) -> GUI for Map2Loop
* [Pybedforms](https://github.com/AndrewAnnex/pybedforms)
* [SA Stratigraphy](https://github.com/RADutchie/SA-Strarigraphy-db) -> Stratigraphy database editor webapp
* [Striplog](https://github.com/agile-geoscience/striplog)
* [Analise_de_Dados_Estruturais_Altamira](https://github.com/fnaghetini/Analise_de_Dados_Estruturais_Altamira/blob/main/Analise_de_Dados_Estruturais_Altamira.ipynb)
* [Global Tectonics](https://github.com/dhasterok/global_tectonics) -> Open source dataset to build on
## Geophysics
* [Geoscience Australia Utilities](https://github.com/RichardScottOZ/geophys_utils)
* [Geophysics for Practicing Geoscientists](https://github.com/geoscixyz/gpg)
* [Potential Field Toolbox](https://github.com/RichardScottOZ/PFToolbox) -> Some xarray based Fast Fourier Transform filters - derivatives, pseudogravity, rpg etc.
  * [Notebook](https://github.com/RichardScottOZ/PFToolbox/blob/master/FFT_Filter.ipynb) -> Class with some examples
 * [Computation geophysics sandbox](https://github.com/yohanesnuwara/computational-geophysics)
 * [RIS Basement Sediment](https://github.com/mdtanker/RIS_basement_sediment) -> Depth to Magnetic Basement in Antarctica
### Electromagnetic
* [Geoscience Australia AEM](https://github.com/GeoscienceAustralia/ga-aem)
* [UH Electromagnetics](https://github.com/jiajiasun/UHElectromagnetics) -> Coursework notebooks on understanding this domain
### Gravity and Magnetics 
* [Harmonica](https://github.com/fatiando/harmonica)
* [Australian Gravity Data](https://github.com/compgeolab/australia-gravity-data)
* [Worms](https://bitbucket.org/fghorow/bsdwormer)
* [Osborne Magnetic](https://github.com/fatiando-data/osborne-magnetic) -> Survey data processing example
### Seismic
* [Segyio](https://github.com/equinor/segyio)
* [Segysak](https://github.com/trhallam/segysak) -> Xarray based seg-y data handling and analysis
### Magnetotellurics
* [MtPy](https://github.com/RichardScottOZ/mtpy)
* [Mineral Stats Toolkit](https://github.com/RichardScottOZ/mineral-stats-toolkit) -> Distant to MT features analaysis
* [mtwaffle](https://github.com/kinverarity1/mtwaffle) -> MT data analysis examples
* [pyMT] (https://github.com/eroots/pyMT)
* [resistics](https://github.com/resistics/resistics)
   
### Gridding
* [PyGMT](https://www.pygmt.org/latest/)
* [Verde](https://github.com/fatiando/verde)
* [Grid_aeromag](https://github.com/rmorel/grid-aeromag) -> Brazilian gridding example
* [Pseudogravity](http://www.cpgg.ufba.br/sato/cursos/geo542/all.f) -> From Blakely, 95
### Inversion
* [SimPEG](https://github.com/RichardScottOZ/simpeg)
  * [SimPEG fork](https://github.com/RichardScottOZ/simpeg)
  * [Transform 2020 SimPEG](https://github.com/simpeg/transform-2020-simpeg)
  * [Transform 2021 SimPEG](https://github.com/RichardScottOZ/transform-2021-simpeg)
  * [SimPEG scripts](https://github.com/fourndo/SimPEG_Scripts)
  * [Astic Joint Inversion example](https://github.com/simpeg-research/Astic-2019-PGI)
* [Gimli](https://github.com/gimli-org/gimli)
* [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x)
* [USGS anonymous ftp](https://pubs.er.usgs.gov/publication/tm7C17) 
* [USGS Software](https://pubs.usgs.gov/of/1995/ofr-95-0077/of-95-77.html) -> longer list of older useful stuff: dosbox, anyone?
* [Geophysics Subroutines[(https://github.com/VictorCarreira/Geophysics) -> Fortran code
* [2020 Aachen Inversion problems](https://github.com/RichardScottOZ/2020-aachen-inverse-problems) -> Overview of gravity inversion theory
    
    
## Geochemistry
* [Pyrolite](https://github.com/morganjwilliams/pyrolite)
* [Levelling](https://github.com/GeoscienceAustralia/geochemical-levelling)
* [Pygeochem tools](https://github.com/RADutchie/pygeochemtools)
* [Geoquimica](https://github.com/gferrsilva/geoquimica)
  
## Drilling
* [dh2loop](https://github.com/Loop3D/dh2loop) -> Drilling Interval assistance
* [PyGSLib](https://github.com/opengeostat/pygslib) -> Downhole surveying and interval normalising

## Remote Sensing
* [Awesome spectral indices](https://github.com/davemlz/awesome-spectral-indices) -> Guide to spectral index creation
* [Open Data Cube](https://www.opendatacube.org/)
  * [DEA Notebooks](https://github.com/GeoscienceAustralia/dea-notebooks) -> Code for use in ODC style workflows
  * [Datacube-stats](https://github.com/daleroberts/datacube-stats) -> Statistical analysis library for ODC
  * [Geo Notebooks](https://github.com/Element84/geo-notebooks) -> Code examples from Element 84
 ## Serverless
 * [Kerchunk](https://github.com/RichardScottOZ/kerchunk) -> Serverless access to cloud based data via Zarr
  * [Kerchunk geoh5](https://github.com/RichardScottOZ/Kerchunk-geoh5) -> Access to Geoscient Analyst/geoh5 projects serverlessly via kerchunk
### Stac catalogues
  * [ODC-Stac](https://github.com/opendatacube/odc-stac) -> Database free Open Data Cube
  * [Intake-stac](https://github.com/intake/intake-stac)
  * [Sat-search](https://github.com/sat-utils/sat-search)
  * [Pystac](https://github.com/stac-utils/pystac)
  * [Stackstac](https://github.com/RichardScottOZ/stackstac) ->  Metadata speeded up dask and xarray timeseries
    * [DEA Stackstac](https://github.com/RichardScottOZ/DEA-stackstac) -> Examples of working with Digital Earth Australia data
### Statistics
  * [Hdstats](https://github.com/RichardScottOZ/hdstats) -> Algorithmic basis of geometric medians
  * [Hdmedians](https://github.com/RichardScottOZ/hdmedians)
### Visualisation
  * [TV](https://github.com/daleroberts/tv) -> view satellite imagery in a terminal
  * [Titiler](https://github.com/developmentseed/titiler-pds)
* [Sits](https://github.com/RichardScottOZ/sits)
* [Hsdar](https://rdrr.io/cran/hsdar/man/hsdar-package.html)
* [Stars](https://r-spatial.github.io/stars/)
## Mineral Potential
* [Nickel Mineral Potential Mapping](https://github.com/RichardScottOZ/Nickel-Mineral-Potential-Modelling) -> ESRI Based analysis  
* [Prospectivity Online Tool](https://github.com/mvalenta100/prospectivity-online-tool)
## Mining Economics
* [Bluecap](https://github.com/RichardScottOZ/bluecap) -> Framework from Monash University for assessing mine viability
* [Zipfs Law](https://github.com/RichardScottOZ/ZipfsLaw_Quadrilatero_Ferrifero) -> Curve fitting the distribution of Mineral Depositions
* [PyASX](https://github.com/jericmac/pyasx) -> ASX Data Feed scraping
* [Metal Price API](https://github.com/chutommy/metal-price) -> Containerised Microservice
## Visualisation 
* [Textbook](https://github.com/rougier/scientific-visualization-book)
* [Napari](https://github.com/napari/napari) -> Multidimensional image viewer
* [Holoviews](https://github.com/holoviz/holoviews) -> Large scale data visualisation
* [Graphviz](https://pygraphviz.github.io/documentation/stable/install.html#windows-install) -> Graph plotting/viewing assistance windows installation info
* [Spatial-kde](https://github.com/mblackgeo/spatial-kde)
* [CET Perceptually Uniform Colormaps](https://github.com/coatless-rpkg/cetcolor)
* [PU Colormaps](https://github.com/thomasostersen/pu_cmaps) -> Formatted for user in Geoscience Analyst

## Geospatial
* [Geospatial](https://github.com/giswqs/geospatial) >- installs multiple common python packages
* [Geospatial python](https://forrest.nyc/75-geospatial-python-and-spatial-data-science-resources-and-guides/) -> Curated list

## PyData Stack
* [Numpy Multidimensional arrays](https://numpy.org/)
* [Pandas Tabular data analysis](https://pandas.pydata.org/)
* [Matplotlib visualisation](https://matplotlib.org/)
* [Zarr](https://github.com/zarr-developers/zarr-python) -> Compressed, chunked distributed arrays
* [Dask](https://github.com/dask/dask) -> Parallel, distributed computing
  *[Dask Cloud Provider](https://github.com/RichardScottOZ/dask-cloudprovider) -> Atuomatica
* [Python Geospatial Ecosystem](https://github.com/loicdtx/python-geospatial-ecosystem) -> Curated information

## C
* [GDAL](https://github.com/OSGeo/gdal) -> Absolutely crucial data transformation and analysis framework

## Data Science
* [Python Data Science Template](https://github.com/RichardScottOZ/python-data-science-template) -> Project package setup
* [Awesome python data science](https://github.com/krzjoa/awesome-python-data-science) -> Curated guide

## Science
* [Awesome scientific computing](https://github.com/nschloe/awesome-scientific-computing)

## Docker
* [AWS Deep Learning Containers](https://github.com/aws/deep-learning-containers)
* [Spatial Docker](https://github.com/stevenpawley/spatial-docker)
* [DL Docker Geospatial](https://github.com/sshuair/dl-docker-geospatial)
* [Rocker](https://github.com/rocker-org/geospatial)
* [Docker Lambda](https://github.com/lambgeo/docker-lambda)
* [Geobase](https://github.com/opendatacube/geobase)
* [DL Docker Geospatial](https://github.com/sshuair/dl-docker-geospatial)
    

# Ontologies
* [Geological Society of Queensland vocabularies](https://github.com/geological-survey-of-queensland/vocabularies)
  * [Geological Properties Database](https://github.com/geological-survey-of-queensland/geological-properties-database)
  * [Geofeatures](https://github.com/geological-survey-of-queensland/geofeatures-ont)
* [Stratigraphic](https://github.com/GeoscienceAustralia/strat-ontology-graph-API)
* [Geoscience Knowledge Manager](https://github.com/Loop3D/GKM)

# Books
* [Machine Learning in the Oil and Gas industry](https://github.com/Apress/machine-learning-oil-gas-industry)
* [Python geospatial analysis cookbook](https://github.com/mdiener21/python-geospatial-analysis-cookbook)
* [Geocomputation with R](https://github.com/Robinlovelace/geocompr)

# Papers
* [EarthArxiv](https://github.com/eartharxiv/API/issues) -> Download papers from the preprint archive


# Other
* [GXPy](https://github.com/GeosoftInc/gxpy) -> Geosoft Python API