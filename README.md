# mineral-exploration-machine-learning
This page lists resources for mineral exploration and machine learning, generally with useful code and examples. 
ML and Data Science is a huge field, these are resources I have found useful and/or interesting to me in practice.
Links currently to a fork of a repository are because I have changed something to use and put in a list for reference.
Resources are also given for data analysis, transformation and visualisation as that is most of the work.

Suggestions welcome: open an issue.

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
* [TorchGeo](https://github.com/microsoft/torchgeo) -> Pytorch library for remote sensing style models
* [Geo Deep Learning](https://github.com/RichardScottOZ/geo-deep-learning) -> Simple deep learning framework based on RGB


# Mineral Prospectivity
## Australia
* [Machine learning for geological mapping : algorithms and applications](https://eprints.utas.edu.au/18571/) -> PhD thesis with code and data
* [Transform 2022 Tutorial](https://github.com/Solve-Geosolutions/transform_2022) -> Random forest example
  * [Video](https://www.youtube.com/watch?v=C4YvnLMzYDc)
* [Tin-Tungsten](https://medium.com/@thomas.ostersen/tin-tungsten-prospecting-with-machine-learning-in-northeast-tasmania-australia-3c23519f81cf)
  * [Collab](https://colab.research.google.com/drive/168PSo21-Jkwdz8xOmr5-rX9_DL3SInCN?usp=sharing)
## Explorer Challenge
* [Explorer Challenge](https://github.com/RichardScottOZ/explore_australia) -> OZ Minerals run competition with Data Science introduction
## Explore SA - South Australian Department of Energy and Mining Competition
* [Winners](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/abba4f54-b6ef-4fe4-b951-57f11299d490) -> SARIG data information
* [Caldera](https://github.com/mrodda117/CalderaPublic) -> Caldera Analytics analysis
* [IncertoData](https://github.com/RichardScottOZ/ExploreSA/tree/master/Data_submission_competition)
* [Butterworth and Barnett](https://github.com/RichardScottOZ/gawler-exploration) -> Butterworth and Barnett entry
* [Data Driven Mineralisation Mapping](https://github.com/Abdallah-M-Ali/mman4020_Data_Driven_Mineralisation_Mapping/blob/main/docs/tensor.ipynb)
## Brazil
* [Mapa Preditivo](https://github.com/fnaghetini/Mapa-Preditivo) -> Brazil student project
* [Mineral Prospectivity Mapping](https://github.com/Eliasmgprado/MineralProspectivityMapping)
* [3D Weights of Evidence](https://github.com/e-farahbakhsh/3DWofE)
* [Geological Complexity SMOTE](https://github.com/Eliasmgprado/GeologicalComplexity_SMOTE)
  * [Paper](https://doi.org/10.1016/j.oregeorev.2020.103611)
* [MPM Jurena](https://github.com/victsnet/MPM---Juruena-Mineral-Province ) -> Jurena Mineral Province
## China
* [MPM by ensemble learning](https://github.com/ZhiqiangZhangCUGB/MPM-by-ensemble-learning) -> Qingchengzi Pb-Zn-Ag-Au polymetallic district China
* [Mineral Prospectivity Prediction Convolutional Neural Networks](https://github.com/yangna815/Mineral-Prospectivity-Prediction-Convolutional-Neural-Networks) -> CNN Example with a few architectures [a paper by this author uses GoogleNet]
* [Mineral Prospectivity Prediction by CSAE](https://github.com/yangna815/Mineral-Prospectivity-Prediction-by-CSAE)
* [Mineral Prospectivity Prediction by CAE](https://github.com/yangna815/Mineral-Prospectivity-Prediction-by-CAE)
## Sudan
* [Mineral Prospectivity Mapping ML](https://github.com/Abdallah-M-Ali/Mineral-Prospectivity-Mapping-ML)

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
* [Automated Well Log Correlation](https://github.com/dudley-fitzgerald/AutomatedWellLogCorrelation)

## Stratigraphy
* [Predicatops](https://github.com/JustinGOSSES/predictatops) -> Stratigraphic predication designed for hydrocarbon

## Geophysics
### Inversion
* [Machine Learning and Geophysical Inversion](https://github.com/ezygeo-ai/machine-learning-and-geophysical-inversion) -> reconstruct paper from Y. Kim and N. Nakata (The Leading Edge, Volume 37, Issue 12, Dec 2018)
### Structure
* [Lineament Learning](https://github.com/aminrd/LineamentLearning) -> Fault prediction and mapping via potential field deep learning and clustering

## Petrophysics
* [ML4Rocks](https://github.com/clberube/ml4rocks) -> Some intro work

# Geochemistry
* [ICBMS Jacobina](https://github.com/gferrsilva/icpms-jacobina) -> Analysis of pyrite chemistry from a gold deposit
* [LewisML](https://github.com/RichardScottOZ/LewisML) -> Analysis of the Lewis Formation
* [Global geochemistry](https://github.com/dhasterok/global_geochemistry)
* [QMineral Modeller](https://github.com/gferrsilva/QMineral_Modeller) -> Mineral Chemistry virtual assistant from the Brazilian geological survey
* [Journal of Geochemical Exploration - Manifold](https://github.com/geometatqueens/2020---Journal-of-Geochemical-Exploration--Manifold)
* [MICA](https://github.com/bluetyson/MICA_shiny) -> Chemical composition, in Shiny
* [Dash Geochemical Prospection](https://github.com/pvabreu7/DashGeochemicalProspection) -> Web-app classifying stream sediments with K-means

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
## Word Embeddings
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
* [Portuguese Word Embeddings](https://github.com/nathanshartmann/portuguese_word_embeddings) 
* [Spanish Word Embeddings](https://github.com/dccuchile/spanish-word-embeddings)

# Remote Sensing
* [CNN Sentinel](https://github.com/jensleitloff/CNN-Sentinel) -> Overview about land-use classification from satellite data with CNNs based on an open dataset
## Spectral Unmixing
* [Hyperspectral Deep Learning Review](https://github.com/mhaut/hyperspectral_deeplearning_review)
* [Hyperspectral Autoencoders](https://github.com/RichardScottOZ/hyperspectral-autoencoders)
* [Deeplearn HSI](https://github.com/hantek/deeplearn_hsi)
* [3DCAE-hyperspectral-classification](https://github.com/MeiShaohui/3DCAE-hyperspectral-classification)
* [DeHIC](https://github.com/jingge326/DeHIC)
* [Pysptools](https://github.com/RichardScottOZ/pysptools) -> also has useful heuristic algorithms
* [Spectral Python](https://github.com/spectralpython/spectral)
* [Spectral Dataset RockSL](https://github.com/RichardScottOZ/spectral-dataset-RockSL) -> Open spectral dataset
* [Unmixing](https://github.com/RichardScottOZ/unmixing)


## Other
* [Network Analysis of Mineralogical Systems](https://github.com/lic10/DTDI-DataAnalysis)
  * [Data](http://www.minsocam.org/MSA/AmMin/TOC/2017/Aug2017_data/AM-17-86104.zip) -> Data from paper here
* [Geoanalytics and machine learning](https://github.com/victsnet/Geoanalytics-and-Machine-Learning)
* [Machine Learning Subsurface](https://github.com/PyBrown/Machine-Learning)
* [ML Geoscience](https://github.com/DIG-Kaust/MLgeoscience)
* [Be a Geoscience Detective](https://github.com/bluetyson/Be-a-geoscience-detective)
* [Earth ML](http://earthml.holoviz.org/tutorial/Machine_Learning.html) -> Some basic tutorials in PyData approaches

## Platforms
* [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) -> Computing platform connected to data sources
## Guides
* [Satellite Image Deep Learning](https://github.com/robmarkcole/satellite-image-deep-learning)
* [Earth Observation](https://github.com/RichardScottOZ/awesome-earthobservation-code)
* [Earth Artificial Intelligence](Awesome-Earth-Artificial-Intelligence)


# Data Quality
* [Geoscience Data Quality for Machine Learning](https://github.com/RichardScottOZ/Geoscience-Data-Quality-for-Machine-Learning) -> Geoscience Data Quality for Machine Learning
* [Australian Gravity Data](https://github.com/RichardScottOZ/australia-gravity-data) -> Overview and analysis of gravity station data
* [Geodiff](https://github.com/MerginMaps/geodiff) -> Comparison of vector datasets
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
  * [kmedoids](https://github.com/kno10/python-kmedoids)
* [Picasso](https://github.com/pachterlab/picasso)
### Bayesian
* [Bayseg](https://github.com/cgre-aachen/bayseg) -> Spatial segmentation
## Explainability
* [InterpretML](https://github.com/interpretml/interpret) -> Interpreting models of tabular data
  * [InterpretML](https://github.com/interpretml/interpret-community) -> Community addition

## Deep Learning
* [Deep Colormap Extraction](https://github.com/RichardScottOZ/deep_colormap_extraction) -> Trying to extract a data scale from pictures
* [Extract and Classify Images from Geoscience Documents](https://github.com/bolgebrygg/extract-and-classify-images-from-geoscience-documents)
### Data
* [Xbatcher](https://github.com/RichardScottOZ/xbatcher) -> Xarray based data reading for deep learning
### Explainability
* [Shap Values](https://github.com/slundberg/shap)

### Self-supervised learning
* [Self Supervised](https://github.com/untitled-ai/self_supervised) -> Pytorch lightning implementations of multiple algorithms
* [Simclr](https://github.com/google-research/simclr)
* [Awesome self-supervised learning](https://github.com/jason718/awesome-self-supervised-learning) -> Curated list
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
  * [T21 Seismic Notebook](https://github.com/stevejpurves/t21-seismic-notebook)
  * [Practical Seismic with Python](https://github.com/gmac161/practical-seismic-t21-tutorial)
  * [Transform 2021 Simpeg](https://github.com/simpeg/transform-2021-simpeg)
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
  * [Stratigraphy](https://en.wikipedia.org/wiki/Stratigraphy) 
* [Geochemistry](https://en.wikipedia.org/wiki/Geochemistry)
* [Geophysics](https://en.wikipedia.org/wiki/Geophysics)
* [Remote Sensing]()

# Web Services
If listed it is assumed they are data generally, if just pictures like WMS it will say so.
## Australia
* [AusGIN](https://www.geoscience.gov.au/web-services)
* [Geoscience Australia](http://services.ga.gov.au/)
  * [Geoscience Australia Catalogue Service](https://ecat.ga.gov.au/geonetwork/srv/eng/csw?request=GetCapabilities&service=CSW&acceptVersions=2.0.2&acceptFormats=application%2Fxml)
* [NSW](https://www.regional.nsw.gov.au/meg/geoscience/products-and-data/gis-web-services)
* [Queensland](https://gisservices.information.qld.gov.au/arcgis/rest/services)
* [SARIG](https://map.sarig.sa.gov.au/MapViewer/StartUp/?siteParams=WebServicesWidget)
* [Tasmania WFS](https://www.mrt.tas.gov.au/products/digital_data/web_feature_service)
* [Victoria Geonetwork](http://geology.data.vic.gov.au/)
* [Western Australia](https://services.slip.wa.gov.au/public/services/SLIP_Public_Services/Industry_and_Mining_WFS/MapServer/WFSServer)
## Brazil
* [Brazil Geoportal](https://geoportal.cprm.gov.br/server/rest/services)
* [Brazil CPRM](https://geoportal.cprm.gov.br/image/rest/services
## Canada
* [NWT](https://services3.arcgis.com/GSr8HAQhtEt4sNnv/arcgis/rest/services/)
## Peru
* [Ingement](https://geocatmin.ingemmet.gob.pe/arcgis/rest/services)
* [Environmental](https://geo.serfor.gob.pe/geoservicios/rest/services)
## Sweden
* [SGU Magnetics WMS](https://resource.sgu.se/service/wms/130/flyggeofysik-magnet)
## Other
* [GTK](https://www.gtk.fi/en/services/data-sets-and-online-services-geo-fi/map-services/) -> Geological Survey of Finland
  * [Finland](https://gtkdata.gtk.fi/arcgis/rest/services)
* [Portugal Geology](https://inspire.lneg.pt/arcgis/rest/services/CartografiaGeologica/CGP1M/MapServer)
* [Spain](https://mapas.igme.es/gis/rest/services)
* [USGS World Mineral](https://mrdata.usgs.gov/services/wfs/ofr20051294?version=1.1.0)
* [Minnesota](https://mngs-umn.opendata.arcgis.com/)
# APIs
* [Open Data API](https://github.com/RichardScottOZ/open-data-api) -> GSQ Open Data Portal API
  * [Geochemistry parsing](https://github.com/geological-survey-of-queensland/geochemistry_parsing)
* [CORE](https://core.ac.uk/data) -> Open Research Texts
  * [API Notebook](https://colab.research.google.com/drive/1_bjqDQhqj7AnSfoCAXDCMOnGZLPQWKfu?usp=sharing) -> Example and fucntions
* [SHARE](https://share.osf.io/discover?q=mineral%20AND%20exploration) -> Open Science API
* [USGS Publications](https://pubs.er.usgs.gov/documentation/web_service_documentation)
* [CROSSREF](https://api.crossref.org/swagger-ui/index.html)
* [ADEPT](https://xdd.wisc.edu/adept/) -> GUI to xDD to search 15M harvested papers

# Data Portals
## Australia
* [SARIG](https://map.sarig.sa.gov.au/) -> South Australia Geological Survey
  * [s3 Reports](Reports and textracted versions in s3 bucket with web interface)
  * [Reports](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/cat0/MSearch;jsessionid=492C6538B64080CE8B13E91C79F8B1BA)
  * [Seismic](https://www.petroleum.sa.gov.au/data-centre/seismic-data)
    * [Seismic downloads](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/segy2d/web/segy/ResultSet?siblingtreeid=e6d6d3af10d149d39ba2141b9d1ce660&sid=6fac787ca578408bad6cfb514eb15498&order=NATIVE%28%27LINE%2Fascend%27%29&rpp=-1&set=1&reorder=1&bclabel=Result+Set) -> One page of links
* [Geoscience Australia Portal](https://portal.ga.gov.au/)
* [Exploring for the Future Portal](https://portal.ga.gov.au//eftf) -> Geoscience Australia web portal with download information
* [Geological Survey of Queensland](https://geoscience.data.qld.gov.au/)
* [GEOVIEW](https://geoview.dmp.wa.gov.au/geoview/?Viewer=GeoView) -> Victorian Geological Survey
* [STRIKE](https://strike.nt.gov.au/wss.html) -> Northern Territory Geological Survey
* [MINVIEW](https://minview.geoscience.nsw.gov.au/) -> New South Wales Geological Survey
## Canada
* [Natural Resources Canada](https://www.nrcan.gc.ca/earth-sciences/geography/atlas-canada/explore-our-data/16892
* [Ontario](https://www.geologyontario.mndm.gov.on.ca/ogsearth.html)
* [Quebec](https://gq.mines.gouv.qc.ca/documents/SIGEOM/TOUTQC/ANG/)
* [British Columbia](https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/british-columbia-geological-survey/publications/digital-geoscience-data)
* [Yukon](https://data.geology.gov.yk.ca/)
## USA
* [Michigan](https://geo.btaa.org/)
* [Earth Explorer](https://earthexplorer.usgs.gov) -> USGS Remote Sensing Data Portal
* [National Map Database](http://ngmdb.usgs.gov/maps/mapview/)
* [ReSci](https://www.sciencebase.gov/catalog/item/4f4e4760e4b07f02db47dfb4) -> Registry of Scientific Collections of the National Geological and Geophysical Data Preservation Program
## Other
* [CPRM](https://www.cprm.gov.br/en/Geology-53) -> Brazil Geological Survey
* [GTK](https://www.gtk.fi/en/services/data-sets-and-online-services-geo-fi/) -> Geological Survey of Finland
* [SGU](https://www.sgu.se/en/products/geological-data/use-data-from-sgu/) -> Swedish Geological Survey
* [NGU](https://www.ngu.no/prospecting/) -> Norway Geological Survey
* [OSF](https://osf.io/) -> Open Science Foundation

## Reports
### Australia
* [Northern Territory GEMIS](https://geoscience.nt.gov.au/gemis)
  * [search](https://www.geoscience.nt.gov.au/gemis/ntgsjspui/simple-search?location=1%2F3&query=&rpp=8000&sort_by=score&order=DESC&submit_search=Update)
  * [example](https://geoscience.nt.gov.au/gemis/ntgsjspui/handle/1/74318)
* [South Australia SARIG](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/cat0/MSearch)
* [Western Australia WAMEX](https://www.dmp.wa.gov.au/Geological-Survey/GSWA-publications-and-maps-1399.aspx)
  * [search](https://wamexsearch.net.au/gswa_es_doco#query-language)
  * [api](https://geodocs.dmirs.wa.gov.au/api//documentlist/10/Report_Ref/A68128)
    * [example](https://geodocs.dmirs.wa.gov.au/Web/documentlist/10/Report_Ref/A68128?TOTALpAGES=)
* [Queensland](https://github.com/RichardScottOZ/open-data-api/blob/master/API-Large_Query_Retrieval_Example-Geochem.ipynb)
* [NSW Digs](https://digs.geoscience.nsw.gov.au)
  * [NSW Digs open](https://digsopen.minerals.nsw.gov.au/digsopen/)
  * [API not public](https://digs.geoscience.nsw.gov.au/solr/select/digs?&q=mr_rin:R00026119)
* [PorterGEO](http://portergeo.com.au/database/index.asp) -> World mineral deposits databases with summary overviews
* [Sustainable Minerals Institute[(https://smi.uq.edu.au/programs) -> Queensland organisation of university affiliated researchers producing datasets and knowledge
### Canada
* [British Columbia](https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/british-columbia-geological-survey/publications/digital-geoscience-data#ARIS)
  * [Search](https://aris.empr.gov.bc.ca/search.asp?mode=request&newsearch=Y) -> Mineral Assessment Reports
  * [Publications](http://webmap.em.gov.bc.ca/mapplace/minpot/Publications_Report.asp) -> Publications
* [Ontario](https://data.ontario.ca/en/dataset/assessment-files) -> Mineral Asssessment Reports
  * [Ontario Publications](https://www.geologyontario.mndm.gov.on.ca/Publications_Description.html)
* [Alberta](https://content.energy.alberta.ca/minerals/abmarsv2/?err=se)
  * [Publications](https://ags.aer.ca/products/all-publications?title=&report-id=&publication_type=All&sort_by=created&sort_order=DESC&page=0)
* [Yukon](https://data.geology.gov.yk.ca/AssessmentReports)
* [Manitoba](https://www.manitoba.ca/iem/mines/assess.html)
* [Newfoundland and Labrador](https://gis.geosurv.gov.nl.ca/minesen/geofiles/)
* [Northwest Territories](https://app.nwtgeoscience.ca/Searching/ReferenceSearch.aspx)
* [Nova Scotia](https://gesner.novascotia.ca/novascan/DocumentQuery.faces)
* [Quebec](https://sigeom.mines.gouv.qc.ca/signet/classes/I1102_index?entt=LG&l=A)
* [Saskatchewan](http://mineral-assessment.saskatchewan.ca/Pages/BasePages/Main.aspx)
  * [Search](http://mineral-assessment.saskatchewan.ca/Pages/BasePages/Main.aspx?UseCase=ExternalSearch)
  * [iMaQs](https://web33.gov.mb.ca/imaqs) -> Integrated Mining and Quarrying System
### USA
* [Arizona](http://repository.azgs.az.gov/)
* [Montana](https://www.mbmg.mtech.edu/mbmgcat/public/ListPublications.asp)
* [Nevada](https://pubs.nbmg.unr.edu/Open-File-Reports-s/1861.htm)
* [New Mexico](https://geoinfo.nmt.edu/publications/index.cfml)
* [Minnesota](https://conservancy.umn.edu/handle/11299/708)
* [Michigan](https://www.michigan.gov/egle/about/organization/oil-gas-and-minerals/oil-and-gas/geological-catalog)
*[json](https://data.michigan.gov/api/views/8zkk-z5n4/rows.json?accessType=DOWNLOAD)
* [Alaska](https://dggs.alaska.gov/pubs)
* [Washington](https://www.dnr.wa.gov/publications/ger_publications_list.pdf)
### Other
* [British Geological Survey NERC](https://nora.nerc.ac.uk)
  * [Mineral Potential](https://www2.bgs.ac.uk/mineralsuk/exploration/potential/mrp.html)
  * [Search](https://nora.nerc.ac.uk/cgi/facet/archive/simple2?screen=XapianSearch&dataset=archive&order=&q=Mineral+AND+exploration&_action_search=Search )
  * [API example](https://nora.nerc.ac.uk/cgi/facet/archive/simple2/export_nerc_JSON.js?screen=XapianSearch&dataset=archive&_action_export=1&output=JSON&exp=0%7C1%7C%7Carchive%7C-%7Cq%3A%3AALL%3AIN%3AMineral+AND+exploration%7C-%7C&n=&cache=)
*[GeoLagret](https://www.sgu.se/en/products/search-tools/geolagret/exploration-reports/) -> Sweden
* [MinData](https://www.mindat.org/mineralindex.php) -> Compilation of rock locations from around the world
* [Mineral Databse](https://rruff.info/ima/) -> Exportable list of minerals with scientific properties and ages
* [NASA](https://www.sti.nasa.gov/research-access/)  
* [ResearchGate](https://www.researchgate.net/) -> Researcher and professional network

# Tools
## GIS
  * [QGIS](https://qgis.org/en/site/) -> GIS Data Visualisation and Analysis Open Source desktop application, has some ML tools : Indispensible for some quick and easy viewing
    * [2D Geology in QGIS](https://github.com/frizatch/2DGeology_in_QGIS) -> Workshop for QGIS NA 2020 introducing geologic maps and cross-sections for students and hobbyists
  * [GRASS](https://github.com/OSGeo/grass) 
## 3D
* [PyVista](https://github.com/pyvista/pyvista) -> VTK wrapping api for great data visualisation and analysis
  * [PVGeo](https://pvgeo.org/index.html)
  * [Pyvista-Xarray](https://github.com/RichardScottOZ/pyvista-xarray) -> Transforming xarray data to VTK 3D painlessly: a great library!
  * [OMFVista](https://github.com/OpenGeoVis/omfvista0 ->Pyvista for Open Mining Format
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
* [Vdeo](https://github.com/marcomusy/vedo)
    
## Geospatial General
* [Python resources for earth science](https://github.com/javedali99/python-resources-for-earth-sciences)
## Vector Data
* [Geopandas](https://geopandas.org/en/stable/)
  * [Dask-geopandas](https://github.com/RichardScottOZ/dask-geopandas)
    * [Tutorial](https://github.com/martinfleis/dask-geopandas-tutorial)
* [SF](https://r-spatial.github.io/sf/)
* [PyESRIDump](https://github.com/RichardScottOZ/pyesridump) -> Library to grab data at scale from ESRI Rest Servers

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
* [OMF](https://github.com/gmggroup/omf) -> Open Mining Format for conversion between things
* [PDF Miner](https://github.com/euske/pdfminer)
* [AEM to seg-y](https://github.com/Neil-Symington/AEM2SEG-Y)
* [ASEG GDF2](https://github.com/kinverarity1/aseg_gdf2)
* [CGG Outfile reader](https://github.com/RichardScottOZ/CGG-Out-Reader)
* [Geosoft Grid to Raster](https://github.com/RichardScottOZ/Geosoft-Grid-to-Raster)
* [GOCAD SG Grid Reader](https://github.com/RichardScottOZ/GOCAD_SG_Grid_Reader)
	* [geomodel-2-3dweb](https://github.com/RichardScottOZ/geomodel-2-3dweb) >- In here they have a method to extract data from binary GOCAD SG Grids
* [Leapfrog Mesh Reader](https://github.com/ThomasMGeo/leapfrogmshreader)
* [VTK to DXF](https://github.com/RichardScottOZ/VTK-to-DXF)

## Geochemistry
* [Pygeochemtools](https://github.com/RADutchie/pygeochemtools) -> library and command line to enable rapid QC and plotting of geochemical data
* [SA Geochemical Maps](https://github.com/GeologicalSurveySouthAustralia/SA-geochemical-maps) -> Data Analysis and plotting of South Australia geochemistry data from the Geological Survey of SA
* [Geochemical levenning](https://github.com/GeoscienceAustralia/geochemical-levelling)
* [Scott Halley's geochemistry tutorial](https://github.com/DinaKlim/Scott-Halley-s-geochemistry-tutorial)
* [Periodic Table](https://github.com/pkienzle/periodictable)
## Geostatistics
* [Geostatspy](https://github.com/GeostatsGuy/GeostatsPy)
* [PyInterpolate](https://github.com/DataverseLabs/pyinterpolate)
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
* [Litholog](https://github.com/rgmyr/litholog)
## Geophysics
* [Geoscience Australia Utilities](https://github.com/RichardScottOZ/geophys_utils)
* [Geophysics for Practicing Geoscientists](https://github.com/geoscixyz/gpg)
* [Potential Field Toolbox](https://github.com/RichardScottOZ/PFToolbox) -> Some xarray based Fast Fourier Transform filters - derivatives, pseudogravity, rpg etc.
  * [Notebook](https://github.com/RichardScottOZ/PFToolbox/blob/master/FFT_Filter.ipynb) -> Class with some examples
 * [Computation geophysics sandbox](https://github.com/yohanesnuwara/computational-geophysics)
 * [RIS Basement Sediment](https://github.com/mdtanker/RIS_basement_sediment) -> Depth to Magnetic Basement in Antarctica
 * [Signal Image Processing](https://github.com/PyBrown/Signal-Image-Processing)
### Electromagnetic
* [Geoscience Australia AEM](https://github.com/GeoscienceAustralia/ga-aem)
* [UH Electromagnetics](https://github.com/jiajiasun/UHElectromagnetics) -> Coursework notebooks on understanding this domain
* [AEM Interpretation](https://github.com/Neil-Symington/aem_interp_dash)
### Gravity and Magnetics 
* [Harmonica](https://github.com/fatiando/harmonica)
* [Australian Gravity Data](https://github.com/compgeolab/australia-gravity-data)
* [Worms](https://bitbucket.org/fghorow/bsdwormer)
* [Osborne Magnetic](https://github.com/fatiando-data/osborne-magnetic) -> Survey data processing example
### Seismic
* [Segyio](https://github.com/equinor/segyio)
* [Segysak](https://github.com/trhallam/segysak) -> Xarray based seg-y data handling and analysis
* [Geophysical notes](https://github.com/aadm/geophysical_notes) -> Seismic data processing
### Magnetotellurics
* [MtPy](https://github.com/RichardScottOZ/mtpy)
* [Mineral Stats Toolkit](https://github.com/RichardScottOZ/mineral-stats-toolkit) -> Distant to MT features analaysis
* [mtwaffle](https://github.com/kinverarity1/mtwaffle) -> MT data analysis examples
* [pyMT] (https://github.com/eroots/pyMT)
* [resistics](https://github.com/resistics/resistics)
   
### Gridding
* [GMT](https://github.com/GenericMappingTools/gmt)
  * [PyGMT](https://www.pygmt.org/latest/)
* [Verde](https://github.com/fatiando/verde)
* [Grid_aeromag](https://github.com/rmorel/grid-aeromag) -> Brazilian gridding example
* [Pseudogravity](http://www.cpgg.ufba.br/sato/cursos/geo542/all.f) -> From Blakely, 95
### Inversion
* [SimPEG](https://github.com/RichardScottOZ/simpeg)
  * [Mira Geoscience Fork](https://github.com/MiraGeoscience/simpeg) -> Used for geoapps
  * [SimPEG fork](https://github.com/RichardScottOZ/simpeg)
  * [Transform 2020 SimPEG](https://github.com/simpeg/transform-2020-simpeg)
  * [Transform 2021 SimPEG](https://github.com/RichardScottOZ/transform-2021-simpeg)
  * [SimPEG scripts](https://github.com/fourndo/SimPEG_Scripts)
  * [Astic Joint Inversion example](https://github.com/simpeg-research/Astic-2019-PGI)
* [Gimli](https://github.com/gimli-org/gimli)
* [Tomofast-x](https://github.com/TOMOFAST/Tomofast-x)
* [USGS anonymous ftp](https://pubs.er.usgs.gov/publication/tm7C17) 
* [USGS Software](https://pubs.usgs.gov/of/1995/ofr-95-0077/of-95-77.html) -> longer list of older useful stuff: dosbox, anyone?
* [Geophysics Subroutines](https://github.com/VictorCarreira/Geophysics) -> Fortran code
* [2020 Aachen Inversion problems](https://github.com/RichardScottOZ/2020-aachen-inverse-problems) -> Overview of gravity inversion theory
    
    
## Geochemistry
* [Pyrolite](https://github.com/morganjwilliams/pyrolite)
  * [gs2021 Pyrolite](https://github.com/morganjwilliams/gs2021-pyrolite)
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
* [Orange](https://orangedatamining.com/) -> Data Mining Gui
* [Hdstats](https://github.com/RichardScottOZ/hdstats) -> Algorithmic basis of geometric medians
* [Hdmedians](https://github.com/RichardScottOZ/hdmedians)
### Visualisation
  * [TV](https://github.com/daleroberts/tv) -> view satellite imagery in a terminal
  * [Titiler](https://github.com/developmentseed/titiler-pds)
* [Sits](https://github.com/RichardScottOZ/sits)
* [Hsdar](https://rdrr.io/cran/hsdar/man/hsdar-package.html)
* [Stars](https://r-spatial.github.io/stars/)
* [Peru Gold Mining SAR](https://gist.github.com/RichardScottOZ/92f4681d9b1931593417eec1001d14bd)
## Mineral Potential
* [Nickel Mineral Potential Mapping](https://github.com/RichardScottOZ/Nickel-Mineral-Potential-Modelling) -> ESRI Based analysis  
* [Prospectivity Online Tool](https://github.com/mvalenta100/prospectivity-online-tool)
## Mining Economics
* [Bluecap](https://github.com/RichardScottOZ/bluecap) -> Framework from Monash University for assessing mine viability
* [Zipfs Law](https://github.com/RichardScottOZ/ZipfsLaw_Quadrilatero_Ferrifero) -> Curve fitting the distribution of Mineral Depositions
* [PyASX](https://github.com/jericmac/pyasx) -> ASX Data Feed scraping
* [Metal Price API](https://github.com/chutommy/metal-price) -> Containerised Microservice
## Visualisation 
* [Napari](https://github.com/napari/napari) -> Multidimensional image viewer
* [Holoviews](https://github.com/holoviz/holoviews) -> Large scale data visualisation
* [Graphviz](https://pygraphviz.github.io/documentation/stable/install.html#windows-install) -> Graph plotting/viewing assistance windows installation info
* [Spatial-kde](https://github.com/mblackgeo/spatial-kde)
### Colormaps
* [CET Perceptually Uniform Colormaps](https://github.com/coatless-rpkg/cetcolor)
* [PU Colormaps](https://github.com/thomasostersen/pu_cmaps) -> Formatted for user in Geoscience Analyst
* [Colormap distortions](https://github.com/mycarta/Colormap-distorsions-Panel-app) -> A Panel app to demonstrate distorsions created by non-perceptual colormaps on geophysical data
* [Ripping Data from Colormpas](https://gist.github.com/kwinkunks/485190adcf3239341d8bebac94de3a2b)
* [Open Geoscience Code Projects](https://softwareunderground.github.io/open_geosciene_code_projects_viz/explore/)

## Geospatial
* [Geospatial](https://github.com/giswqs/geospatial) >- installs multiple common python packages
* [Geospatial python](https://forrest.nyc/75-geospatial-python-and-spatial-data-science-resources-and-guides/) -> Curated list

## PyData Stack
* [Anaconda](https://www.anaconda.com/products/distribution) -> Get lots installed already with this package manager.
  *[GDAL et al](https://www.anaconda.com/products/distribution) -> Take the pain out of GDAL and Tensorflow installs here
* [Numpy Multidimensional arrays](https://numpy.org/)
* [Pandas Tabular data analysis](https://pandas.pydata.org/)
* [Matplotlib visualisation](https://matplotlib.org/)
* [Zarr](https://github.com/zarr-developers/zarr-python) -> Compressed, chunked distributed arrays
* [Dask](https://github.com/dask/dask) -> Parallel, distributed computing
  * [Dask Cloud Provider](https://github.com/RichardScottOZ/dask-cloudprovider) -> Automatically start dask clusters on the cloud
  * [Dask Median](https://gist.github.com/andrewdhicks/d89849997453cdfad6fa568816ca7160) -> Notebook giving a Dask median function prototype
* [Python Geospatial Ecosystem](https://github.com/loicdtx/python-geospatial-ecosystem) -> Curated information

## C
* [GDAL](https://github.com/OSGeo/gdal) -> Absolutely crucial data transformation and analysis framework
  * [Tools]() -> Note has many command line tools that are very useful as well

## Data Science
* [Python Data Science Template](https://github.com/RichardScottOZ/python-data-science-template) -> Project package setup
* [Awesome python data science](https://github.com/krzjoa/awesome-python-data-science) -> Curated guide

## Science
* [Python resources for earth sciences](https://github.com/javedali99/python-resources-for-earth-sciences)
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
* [Textbook](https://github.com/rougier/scientific-visualization-book)
* [Machine Learning in the Oil and Gas industry](https://github.com/Apress/machine-learning-oil-gas-industry)
* [Python geospatial analysis cookbook](https://github.com/mdiener21/python-geospatial-analysis-cookbook)
* [Geocomputation with R](https://github.com/Robinlovelace/geocompr)
* [Earthdata Cloud Cookbook](https://github.com/NASA-Openscapes/earthdata-cloud-cookbook) -> How to access NASA resources
* [Data Cleaner's Cookbook](https://www.datafix.com.au/cookbook/about.html) -> Putting unix tools to good use for data wrangling and cleaning

# Other
* [GXPy](https://github.com/GeosoftInc/gxpy) -> Geosoft Python API
* [EarthArxiv](https://github.com/eartharxiv/API/issues) -> Download papers from the preprint archive
* [Essoar](https://www.essoar.org/) -> Preprint paper archive


# Papers with Code 
### NLP
- https://www.sciencedirect.com/science/article/pii/S2590197422000064?via%3Dihub#bib20- -> Geoscience language models and their intrinsic evaluation -> NRCan code above [includes model]
- https://www.researchgate.net/publication/334507958_Word_embeddings_for_application_in_geosciences_development_evaluation_and_examples_of_soil-related_concepts -> GeoVec [includes model]
- https://www.researchgate.net/publication/347902344_Portuguese_word_embeddings_for_the_oil_and_gas_industry_Development_and_evaluation -> PetroVec [includes model]
### Other
- https://www.researchgate.net/publication/318839364_Network_analysis_of_mineralogical_systems

# Papers with Features Data
These you can reproduce the output geospatially from the data given.
### Prospectivity
- https://www.sciencedirect.com/science/article/pii/S016913682100010X#s0135 -> Prospectivity modelling of Canadian magmatic Ni (±Cu ± Co ± PGE) sulphide mineral systems [well worth reading]
- https://www.sciencedirect.com/science/article/pii/S0169136821006612#b0510 -> Data–driven prospectivity modelling of sediment–hosted Zn–Pb mineral systems and their critical raw materials [well worth reading]

# Geospatial Output - No Code
- https://geoscience.data.qld.gov.au/report/cr113697) -> NWMP Data-Driven Mineral Exploration And Geological Mapping

# Journals
- https://www.sciencedirect.com/journal/artificial-intelligence-in-geosciences -> Artificial Intelligence in Geosciences

# Papers - Generally Not ML, or no Code/Data and sometimes no availability at all
- Eventually will separate out into things that have data packages or not like NSW Zone studies.
- However, if interested in an area you can often georeference a picture if nothing else as a rough guide.
- Generally these are not reproducible - a few like the NSW prospectivity zone studies and NWQMP are with some work.  
- Occasional paper here may be listed above


## New to File
- https://www.researchgate.net/project/Bayesian-Machine-Learning-for-Geological-Modeling-and-Geophysical-Segmentation
- https://www.researchgate.net/publication/348983384_Mineral_prospectivity_mapping_using_a_VNet_convolutional_neural_network
- https://www.researchgate.net/publication/355467413_Harnessing_the_Power_of_Artificial_Intelligence_and_Machine_Learning_in_Mineral_Exploration-Opportunities_and_Cautionary_Notes
- https://www.researchgate.net/publication/352251078_Data_Analysis_Methods_for_Prospectivity_Modelling_as_applied_to_Mineral_Exploration_Targeting_State-of-the-Art_and_Outlook
- https://www.researchgate.net/publication/361230503_What_is_Mineral_Informatics

### Australia
- https://www.researchgate.net/publication/352310314_Central_Lachlan_Mineral_Potential_Study
- https://link.springer.com/article/10.1007/s11004-021-09989-z
- https://www.researchgate.net/publication/353253570_A_Truly_Spatial_Random_Forests_Algorithm_for_Geoscience_Data_Analysis_and_Modelling
- https://www.researchgate.net/publication/362260616_Assessing_the_impact_of_conceptual_mineral_systems_uncertainty_on_prospectivity_predictions
- https://www.researchgate.net/publication/353058758_Using_Machine_Learning_to_Map_Western_Australian_Landscapes_for_Mineral_Exploration 
### Brazil
- https://www.researchgate.net/publication/360055592_Predicting_mineralization_and_targeting_exploration_criteria_based_on_machine-learning_in_the_Serra_de_Jacobina_quartz-pebble-metaconglomerate_Au-U_deposits_Sao_Francisco_Craton_Brazil
- https://www.researchgate.net/publication/362263694_Machine_Learning_Methods_for_Quantifying_Uncertainty_in_Prospectivity_Mapping_of_Magmatic-Hydrothermal_Gold_Deposits_A_Case_Study_from_Juruena_Mineral_Province_Northern_Mato_Grosso_Brazil
#### Fuzzy
- https://www.researchgate.net/publication/356508827_Geophysical-spatial_Data_Modeling_using_Fuzzy_Logic_Applied_to_Nova_Aurora_Iron_District_Northern_Minas_Gerais_State_Southeastern_Brazil
- https://www.researchgate.net/publication/348823482_Combining_fuzzy_analytic_hierarchy_process_with_concentration-area_fractal_for_mineral_prospectivity_mapping_A_case_study_involving_Qinling_orogenic_belt_in_central_China
- https://www.researchgate.net/publication/360386350_Application_of_Fuzzy_Gamma_Operator_to_Generate_Mineral_Prospectivity_Mapping_for_Cu-Mo_Porphyry_Deposits_Case_Study_Kighal-Bourmolk_Area_Northwestern_Iran
- https://www.researchgate.net/publication/356937528_Mineral_prospectivity_mapping_a_potential_technique_for_sustainable_mineral_exploration_and_mining_activities_-_a_case_study_using_the_copper_deposits_of_the_Tagmout_basin_Morocco
### Canada
- https://www.researchgate.net/publication/352046255_Study_of_the_Influence_of_Non-Deposit_Locations_in_Data-Driven_Mineral_Prospectivity_Mapping_A_Case_Study_on_the_Iskut_Project_in_Northwestern_British_Columbia_Canada
- https://www.researchgate.net/publication/348111963_Support_Vector_Machine_and_Artificial_Neural_Network_Modelling_of_Orogenic_Gold_Prospectivity_Mapping_in_the_Swayze_greenstone_belt_Ontario_Canada
- https://data.geology.gov.yk.ca/Reference/95936#InfoTab -> Updates to the Yukon Geological Survey’s mineral potential mapping methodology
### China
- https://www.researchgate.net/publication/350788828_Geochemically_Constrained_Prospectivity_Mapping_Aided_by_Unsupervised_Cluster_Analysis
- https://www.researchgate.net/publication/357685352_Determination_of_Predictive_Variables_in_Mineral_Prospectivity_Mapping_Using_Supervised_and_Unsupervised_Methods
- https://www.researchgate.net/publication/347344551_A_positive_and_unlabeled_learning_algorithm_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/361589587_Unlabeled_Sample_Selection_for_Mineral_Prospectivity_Mapping_by_Semi-supervised_Support_Vector_Machine
- https://www.researchgate.net/publication/352983697_Mineral_prospectivity_mapping_by_deep_learning_method_in_Yawan-Daqiao_area_Gansu
- https://www.researchgate.net/publication/360333702_Ensemble_learning_models_with_a_Bayesian_optimization_algorithm_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/353421842_A_hybrid_logistic_regression_gene_expression_programming_model_and_its_application_to_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/354132594_A_Convolutional_Neural_Network_of_GoogLeNet_Applied_in_Mineral_Prospectivity_Prediction_Based_on_Multi-source_Geoinformation
- https://www.researchgate.net/publication/339821823_A_Monte_Carlo-based_framework_for_risk-return_analysis_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/360028637_Three-Dimensional_Mineral_Prospectivity_Mapping_by_XGBoost_Modeling_A_Case_Study_of_the_Lannigou_Gold_Deposit_China
- https://www.researchgate.net/publication/352476625_Machine_Learning-Based_3D_Modeling_of_Mineral_Prospectivity_Mapping_in_the_Anqing_Orefield_Eastern_China
- https://www.researchgate.net/publication/351649498_Mineral_Prospectivity_Mapping_based_on_Isolation_Forest_and_Random_Forest_Implication_for_the_Existence_of_Spatial_Signature_of_Mineralization_in_Outliers
- https://www.researchgate.net/publication/352703015_Data-driven_based_logistic_function_and_prediction-area_plot_for_mineral_prospectivity_mapping_a_case_study_from_the_eastern_margin_of_Qinling_orogenic_belt_central_China
- https://www.researchgate.net/publication/352983697_Mineral_prospectivity_mapping_by_deep_learning_method_in_Yawan-Daqiao_area_Gansu
- https://www.researchgate.net/publication/350817877_Mineral_Prospectivity_Prediction_via_Convolutional_Neural_Networks_Based_on_Geological_Big_Data
- https://www.researchgate.net/publication/350817136_3D_Mineral_Prospectivity_Mapping_Based_on_Deep_Metallogenic_Prediction_Theory_A_Case_Study_of_the_Lala_Copper_Mine_Sichuan_China
- https://www.researchgate.net/publication/350788828_Geochemically_Constrained_Prospectivity_Mapping_Aided_by_Unsupervised_Cluster_Analysis
- https://www.researchgate.net/publication/349034539_A_Comparative_Study_of_Machine_Learning_Models_with_Hyperparameter_Optimization_Algorithm_for_Mapping_Mineral_Prospectivity
- https://www.researchgate.net/publication/358528670_Mineral_Prospectivity_Mapping_Based_on_Wavelet_Neural_Network_and_Monte_Carlo_Simulations_in_the_Nanling_W-Sn_Metallogenic_Province
- https://www.researchgate.net/publication/355749736_Mineral_prospectivity_mapping_using_a_joint_singularity-based_weighting_method_and_long_short-term_memory_network
- https://www.researchgate.net/publication/357584076_Mapping_prospectivity_for_regolith-hosted_REE_deposits_via_convolutional_neural_network_with_generative_adversarial_network_augmented_data
- https://www.researchgate.net/publication/344303914_Random-Drop_Data_Augmentation_of_Deep_Convolutional_Neural_Network_for_Mineral_Prospectivity_Mapping
- https://www.sciencedirect.com/science/article/pii/S2666544121000253 - Microleveling aerogeophysical data using deep convolutional network and MoG-RPCA]

### Eritrea
- https://www.researchgate.net/publication/349158008_Mapping_gold_mineral_prospectivity_based_on_weights_of_evidence_method_in_southeast_Asmara_Eritrea
### Finland
- https://www.researchgate.net/publication/360661926_Target-scale_prospectivity_modeling_for_gold_mineralization_within_the_Rajapalot_Au-Co_project_area_in_northern_Fennoscandian_Shield_Finland_Part_2_Application_of_self-organizing_maps_and_artificial_n

### Greenland
- https://www.researchgate.net/publication/360970965_Identification_of_Radioactive_Mineralized_Lithology_and_Mineral_Prospectivity_Mapping_Based_on_Remote_Sensing_in_High-Latitude_Regions_A_Case_Study_on_the_Narsaq_Region_of_Greenland

### Iran
- https://www.researchgate.net/publication/330359897_Application_of_hybrid_AHP-TOPSIS_method_for_prospectivity_modeling_of_Cu_porphyry_in_Varzaghan_district_Iran
- https://www.researchgate.net/publication/349957803_Regional-Scale_Mineral_Prospectivity_Mapping_Support_Vector_Machines_and_an_Improved_Data-Driven_Multi-criteria_Decision-Making_Technique
- https://www.researchgate.net/publication/321076980_Spatial_analyses_of_exploration_evidence_data_to_model_skarn-type_copper_prospectivity_in_the_Varzaghan_district_NW_Iran
- https://www.researchgate.net/publication/349957803_Regional-Scale_Mineral_Prospectivity_Mapping_Support_Vector_Machines_and_an_Improved_Data-Driven_Multi-criteria_Decision-Making_Technique
- https://www.researchgate.net/publication/333199619_Incorporation_of_principal_component_analysis_geostatistical_interpolation_approaches_and_frequency-space-based_models_for_portraying_the_Cu-Au_geochemical_prospects_in_the_Feizabad_district_NW_Iran
- https://www.researchgate.net/publication/339153591_Sensitivity_analysis_of_prospectivity_modeling_to_evidence_maps_Enhancing_success_of_targeting_for_epithermal_gold_Takab_district_NW_Iran
- https://www.researchgate.net/publication/356872819_Application_of_self-organizing_map_SOM_and_K-means_clustering_algorithms_for_portraying_geochemical_anomaly_patterns_in_Moalleman_district_NE_Iran
- https://www.researchgate.net/publication/348482539_A_new_strategy_for_spatial_predictive_mapping_of_mineral_prospectivity_Automated_hyperparameter_tuning_of_random_forest_approach
- https://www.researchgate.net/publication/352251016_A_simulation-based_framework_for_modulating_the_effects_of_subjectivity_in_greenfield_Mineral_Prospectivity_Mapping_with_geochemical_and_geological_data
- https://www.researchgate.net/publication/351750324_A_data_augmentation_approach_to_XGboost-based_mineral_potential_mapping_An_example_of_carbonate-hosted_Zn_Pb_mineral_systems_of_Western_Iran
- https://www.researchgate.net/publication/361717490_Quantifying_Uncertainties_Linked_to_the_Diversity_of_Mathematical_Frameworks_in_Knowledge-Driven_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/356580903_Evidential_data_integration_to_produce_porphyry_Cu_prospectivity_map_using_a_combination_of_knowledge_and_data_driven_methods
- https://www.researchgate.net/publication/348500913_A_new_strategy_for_spatial_predictive_mapping_of_mineral_prospectivity
- https://www.researchgate.net/publication/353761696_Assessing_the_effects_of_mineral_systems-derived_exploration_targeting_criteria_for_Random_Forests-based_predictive_mapping_of_mineral_prospectivity_in_Ahar-Arasbaran_area_Iran
- https://www.researchgate.net/publication/358507255_A_Comparative_Study_of_Convolutional_Neural_Networks_and_Conventional_Machine_Learning_Models_for_Lithological_Mapping_Using_Remote_Sensing_Data
- https://www.researchgate.net/publication/361529867_Prospectivity_mapping_of_orogenic_lode_gold_deposits_using_fuzzy_models_a_case_study_of_Saqqez_area_NW_of_Iran
- https://www.researchgate.net/publication/351965039_Intelligent_geochemical_exploration_modeling_using_multiclass_support_vector_machine_and_integration_it_with_continuous_genetic_algorithm_in_Gonabad_region_Khorasan_Razavi_Iran
- https://www.researchgate.net/publication/353982380_Porphyry_Cu-Au_prospectivity_modelling_using_semi-supervised_learning_algorithm_in_Dehsalm_district_eastern_Iran_In_Farsi_with_extended_English_abstract
- https://www.researchgate.net/publication/358567148_Applications_of_data_augmentation_in_mineral_prospectivity_prediction_based_on_convolutional_neural_networks
- https://www.researchgate.net/publication/356660905_Deep_GMDH_Neural_Networks_for_Predictive_Mapping_of_Mineral_Prospectivity_in_Terrains_Hosting_Few_but_Large_Mineral_Deposits

### India
- https://www.researchgate.net/publication/355397149_Gold_Prospectivity_Mapping_in_the_Sonakhan_Greenstone_Belt_Central_India_A_Knowledge-Driven_Guide_for_Target_Delineation_in_a_Region_of_Low_Exploration_Maturity
### Phillipines
- https://www.researchgate.net/publication/356546133_Mineral_Prospectivity_Mapping_via_Gated_Recurrent_Unit_Model
### Russia
- https://www.researchgate.net/publication/358431343_Application_of_Maximum_Entropy_for_Mineral_Prospectivity_Mapping_in_Heavily_Vegetated_Areas_of_Greater_Kurile_Chain_with_Landsat_8_Data
- https://www.researchgate.net/publication/354000754_Mineral_Prospectivity_Mapping_for_Forecasting_Gold_Deposits_in_the_Central_Kolyma_Region_North-East_Russia
### South Africa
- https://www.researchgate.net/publication/359294267_Data-driven_multi-index_overlay_gold_prospectivity_mapping_using_geophysical_and_remote_sensing_datasets
- https://www.researchgate.net/publication/361526053_Mineral_prospectivity_mapping_of_gold-base_metal_mineralisation_in_the_Sabie-Pilgrim%27s_Rest_area_Mpumalanga_Province_South_Africa
### Spain
- https://www.researchgate.net/publication/356639977_Machine_learning_models_for_Hg_prospecting_in_the_Almaden_mining_district


## Public Mineral Prospectivity Mapping
Last edited: 29/09/2020
### General
- https://www.researchgate.net/publication/330077321_An_Improved_Data-Driven_Multiple_Criteria_Decision-Making_Procedure_for_Spatial_Modeling_of_Mineral_Prospectivity_Adaption_of_Prediction-Area_Plot_and_Logistic_Functions
- https://www.researchgate.net/publication/272494576_Geological_knowledge_discovery_and_minerals_targeting_from_regolith_using_a_machine_learning_approach
- https://www.researchgate.net/publication/337650865_A_combinative_knowledge-driven_integration_method_for_integrating_geophysical_layers_with_geological_and_geochemical_datasets
- https://www.researchgate.net/publication/235443297_Addressing_challenges_with_exploration_datasets_to_generate_usable_mineral_potential_maps
- https://www.researchgate.net/publication/235443294_The_effect_of_map-scale_on_geological_complexity
- https://www.researchgate.net/publication/235443305_The_effect_of_map_scale_on_geological_complexity_for_computer-aided_exploration_targeting
- https://www.researchgate.net/publication/273284693_Spatial-Contextual_Supervised_Classifiers_Explored_A_Challenging_Example_of_Lithostratigraphy_Classification
- https://www.researchgate.net/publication/267927728_Data-Driven_Evidential_Belief_Modeling_of_Mineral_Potential_Using_Few_Prospects_and_Evidence_with_Missing_Values
- https://www.researchgate.net/publication/273500012_Prediction-area_P-A_plot_and_C-A_fractal_analysis_to_classify_and_evaluate_evidential_maps_for_mineral_prospectivity_modeling
- https://www.researchgate.net/publication/280013864_Geometric_average_of_spatial_evidence_data_layers_A_GIS-based_multi-criteria_decision-making_approach_to_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/220163838_Objective_selection_of_suitable_unit_cell_size_in_data-driven_modeling_of_mineral_prospectivity
- https://www.researchgate.net/publication/229792860_From_Predictive_Mapping_of_Mineral_Prospectivity_to_Quantitative_Estimation_of_Number_of_Undiscovered_Prospects
- https://www.researchgate.net/publication/297777471_Spatial_mathematical_models_for_mineral_potential_mapping
- https://www.researchgate.net/publication/220164155_Support_vector_machine_A_tool_for_mapping_mineral_prospectivity
- https://www.researchgate.net/publication/220164234_Mapping_complexity_of_spatial_distribution_of_faults_using_fractal_and_multifractal_models_Vectoring_towards_exploration_targets
- https://www.researchgate.net/publication/220164488_Geocomputation_of_mineral_exploration_targets
- https://www.researchgate.net/publication/229714681_Classifiers_for_Modeling_of_Mineral_Potential
- https://api.research-repository.uwa.edu.au/portalfiles/portal/5263287/Lysytsyn_Volodymyr_2015.pdf (PhD thesis) GIS-based epithermal copper prospectivity mapping of the Mt Isa Inlier, Australia: Implications for exploration targeting

## Overview
- https://www.researchgate.net/publication/317319129_Natural_Resources_Research_Publications_on_Geochemical_Anomaly_and_Mineral_Potential_Mapping_and_Introduction_to_the_Special_Issue_of_Papers_in_These_Fields
- https://www.researchgate.net/publication/275338029_Introduction_to_the_Special_Issue_GIS-based_mineral_potential_modelling_and_geological_data_analyses_for_mineral_exploration
- https://www.researchgate.net/publication/341472154_Geodata_Science-Based_Mineral_Prospectivity_Mapping_A_Review
- https://www.researchgate.net/publication/339074334_Introduction_to_the_special_issue_on_spatial_modelling_and_analysis_of_ore-forming_processes_in_mineral_exploration_targeting
- https://www.researchgate.net/publication/284890591_Geochemical_Anomaly_and_Mineral_Prospectivity_Mapping_in_GIS
- https://www.researchgate.net/publication/46696293_Selection_of_coherent_deposit-type_locations_and_their_application_in_data-driven_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/331852267_Applying_Spatial_Prospectivity_Mapping_to_Exploration_Targeting_Fundamental_Practical_issues_and_Suggested_Solutions_for_the_Future

## Geochemistry
- https://www.researchgate.net/publication/333497470_Integration_of_auto-encoder_network_with_density-based_spatial_clustering_for_geochemical_anomaly_detection_for_mineral_exploration
- https://www.researchgate.net/publication/277813662_Supervised_Geochemical_Anomaly_Detection_by_Pattern_Recognition
- https://www.researchgate.net/publication/331505001_Deep_learning_and_its_application_in_geochemical_mapping
- https://www.researchgate.net/publication/319303831_Introduction_to_the_thematic_issue_Analysis_of_exploration_geochemical_data_for_mapping_of_anomalies
- https://www.researchgate.net/publication/259716832_Supervised_and_unsupervised_classification_of_near-mine_soil_Geochemistry_and_Geophysics_data
- https://www.researchgate.net/publication/238505045_Analysis_and_mapping_of_geochemical_anomalies_using_logratio-transformed_stream_sediment_data_with_censored_values
- https://www.researchgate.net/publication/321275541_Weighting_stream_sediment_geochemical_samples_as_exploration_indicator_of_deposit_-_type
- https://www.researchgate.net/publication/220164381_Application_of_geochemical_zonality_coefficients_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/257026525_Primary_geochemical_characteristics_of_mineral_deposits_-_Implications_for_exploration
- https://www.researchgate.net/publication/257189047_Geochemical_mineralization_probability_index_GMPI_A_new_approach_to_generate_enhanced_stream_sediment_geochemical_evidential_map_for_increasing_probability_of_success_in_mineral_potential_mapping
- https://www.researchgate.net/publication/272091723_Geochemical_characteristics_of_mineral_deposits_Implications_for_ore_genesis
- https://www.researchgate.net/publication/249544991_Usefulness_of_stream_order_to_detect_stream_sediment_geochemical_anomalies
- 
## Fuzzy
- https://www.researchgate.net/publication/267816279_Fuzzification_of_continuous-value_spatial_evidence_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/272170968_A_Comparative_Analysis_of_Weights_of_Evidence_Evidential_Belief_Functions_and_Fuzzy_Logic_for_Mineral_Potential_Mapping_Using_Incomplete_Data_at_the_Scale_of_Investigation
- https://www.researchgate.net/publication/301635716_Union_score_and_fuzzy_logic_mineral_prospectivity_mapping_using_discretized_and_continuous_spatial_evidence_values
- 
## Uncertainty
- https://www.researchgate.net/publication/255909185_The_upside_of_uncertainty_Identification_of_lithology_contact_zones_from_airborne_geophysics_and_satellite_data_using_random_forests_and_support_vector_machines
- https://www.researchgate.net/publication/333339659_Incorporating_conceptual_and_interpretation_uncertainty_to_mineral_prospectivity_modelling
- https://www.researchgate.net/publication/235443307_Managing_uncertainty_in_exploration_targeting
 
## Geospatial Maps
### Australia
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/83884 - Nickel PGE
- https://www.researchgate.net/publication/334440382_Mapping_iron_oxide_Cu-Au_IOCG_mineral_potential_in_Australia_using_a_knowledge-driven_mineral_systems-based_approach
#### South Australia
- https://www.researchgate.net/publication/335313790_Prospectivity_modelling_of_the_Olympic_Cu-Au_Province - https://services.sarig.sa.gov.au/raster/ProspectivityModelling/wms?service=wms&version=1.1.1&REQUEST=GetCapabilities
- An assessment of the uranium and geothermal prospectivity of east-central South Australia - https://d28rz98at9flks.cloudfront.net/72666/Rec2011_034.pdf
#### QLD
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/130587 - Tennant Creek - Mt Isa
- Navigate to: qdexdata.dnrme.qld.gov.au Enter the report number: 113697 - NWMP Data Driven Mineral Exploration and Geological Mapping
#### NT
- https://www.researchgate.net/publication/285235798_An_assessment_of_the_uranium_and_geothermal_prospectivity_of_the_southern_Northern_Territory
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/76423 Iron oxide-copper-gold potential of the southern Arunta Region
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/74288 Southern Northern Territory uranium and geothermal energy systems assessment digitil data package
#### WA
- https://www.sciencedirect.com/science/article/abs/pii/S0301926810002111 - Yilgarn
- https://researchdata.edu.au/predictive-mineral-discovery-gold-mineral/1209568?source=suggested_datasets - Predictive mineral discovery in the eastern Yilgarn Craton: an example of district-scale targeting of an orogenic gold mineral system - https://d28rz98at9flks.cloudfront.net/82617/Y4_Gold_Targeting.zip
- http://dmpbookshop.eruditetechnologies.com.au/product/prospectivity-analysis-of-the-halls-creek-orogen-western-australia-using-a-mineral-systems-approach-geographical-product-n15af3zp.do
- http://dmpbookshop.eruditetechnologies.com.au/product/mineral-prospectivity-of-the-king-leopold-orogen-and-lennard-shelf-analysis-of-potential-field-data-in-the-west-kimberley-region-geographical-product-n14bnzp.do
- http://dmpbookshop.eruditetechnologies.com.au/product/mineral-systems-analysis-of-the-west-musgrave-province-regional-structure-and-prospectivity-modelling-geographical-product-n12dzp.do
- http://dmpbookshop.eruditetechnologies.com.au/product/mineral-systems-analysis-of-the-west-musgrave-province-regional-structure-and-prospectivity-modelling.do  $22 purchase
- http://dmpbookshop.eruditetechnologies.com.au/product/regional-scale-targeting-for-gold-in-the-yilgarn-craton-part-1-of-the-yilgarn-gold-exploration-targeting-atlas.do $55 purchase
- http://dmpbookshop.eruditetechnologies.com.au/product/district-scale-targeting-for-gold-in-the-yilgarn-craton-part-2-of-the-yilgarn-gold-exploration-targeting-atlas.do$55 purchase
- https://researchdata.edu.au/prospectivity-analysis-using-063-m436/1424743 - Prospectivity analysis using a mineral systems approach - Capricorn case study project CSIRO Prospectivity analysis using a mineral systems approach - Capricorn case study project (13.5 GB Download)
- https://www.researchgate.net/publication/263928515_Towards_Australian_metallogenic_maps_through_space_and_time
- https://www.researchgate.net/publication/273073675_Building_a_machine_learning_classifier_for_iron_ore_prospectivity_in_the_Yilgarn_Craton
#### NSW
- https://www.resourcesandgeoscience.nsw.gov.au/miners-and-explorers/geoscience-information/projects/mineral-potential-mapping#_southern-_new-_england-_orogen-mineral-potential
- https://search.geoscience.nsw.gov.au/product/9233 - 54 Curnamona Province and Delamerian-Thomson Orogen
- https://search.geoscience.nsw.gov.au/product/9253 - Eastern Lachlan Orogen https://www.smedg.org.au/GSNSW_2019_Blevin.pdf
- 
### Brazil
- https://www.researchgate.net/publication/340633563_CATALOG_OF_PROSPECTIVITY_MAPS_OF_SELECTED_AREAS_FROM_BRAZIL
- https://www.researchgate.net/publication/341936771_Modeling_of_Cu-Au_Prospectivity_in_the_Carajas_mineral_province_Brazil_through_Machine_Learning_Dealing_with_Imbalanced_Training_Data
- https://www.scielo.br/scielo.php?script=sci_arttext&pid=S2317-48892016000200261 - Sao Francisco Craton Nickel
- https://www.researchgate.net/publication/287270273_Nickel_prospective_modelling_using_fuzzy_logic_on_nova_Brasilandia_metasedimentary_belt_Rondonia_Brazil
### Australia
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/83884 - Nickel PGE
- https://www.researchgate.net/publication/334440382_Mapping_iron_oxide_Cu-Au_IOCG_mineral_potential_in_Australia_using_a_knowledge-driven_mineral_systems-based_approach
- https://researchdata.edu.au/predictive-model-opal-mining-approach/673159/?refer_q=rows=15/sort=score%20desc/class=collection/p=2/q=mineral%20prospectivity%20map/ - Opal
- https://www.researchgate.net/publication/248211737_A_continent-wide_study_of_Australia's_uranium_potential
- https://www.sciencedirect.com/science/article/abs/pii/S0169136821002250?via%3Dihub
### SA
- https://www.researchgate.net/publication/335313790_Prospectivity_modelling_of_the_Olympic_Cu-Au_Province - https://services.sarig.sa.gov.au/raster/ProspectivityModelling/wms?service=wms&version=1.1.1&REQUEST=GetCapabilities
- http://www.energymining.sa.gov.au/minerals/knowledge_centre/mesa_journal/previous_feature_articles/new_prospectivity_map
- https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/e59cd4ba-1a0a-4911-9e6a-58d80576678d - Olympic Domain IOCG Prospectivity model
- An assessment of the uranium and geothermal prospectivity of east-central South Australia - https://d28rz98at9flks.cloudfront.net/72666/Rec2011_034.pdf
- https://data.gov.au/dataset/ds-ga-a8619169-1c2a-6697-e044-00144fdd4fa6/details?q= 
- https://www.pir.sa.gov.au/__data/assets/pdf_file/0011/239636/204581-001_wise_high.pdf - Eastern Gawler - WPA
### QLD
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/130587 - Tennant Creek - Mt Isa
- Navigate to: qdexdata.dnrme.qld.gov.au Enter the report number: 113697 - NWMP Data Driven Mineral Exploration and Geological Mapping
### WA
- https://www.sciencedirect.com/science/article/abs/pii/S0301926810002111 - Yilgarn Karol Czarnota
- https://researchdata.edu.au/predictive-mineral-discovery-gold-mineral/1209568?source=suggested_datasets - Predictive mineral discovery in the eastern Yilgarn Craton: an example of district-scale targeting of an orogenic gold mineral system - https://d28rz98at9flks.cloudfront.net/82617/Y4_Gold_Targeting.zip
- https://www.researchgate.net/publication/229333177_Prospectivity_analysis_of_the_Plutonic_Marymia_Greenstone_Belt_Western_Australia
- https://www.researchgate.net/publication/280039091_Mineral_systems_approach_applied_to_GIS-based_2D-prospectivity_modelling_of_geological_regions_Insights_from_Western_Australia
- https://www.researchgate.net/publication/351238658_Understanding_Ore-Forming_Conditions_using_Machine_Reading_of_Text
### NT
- https://www.researchgate.net/publication/285235798_An_assessment_of_the_uranium_and_geothermal_prospectivity_of_the_southern_Northern_Territory
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/76423 Iron oxide-copper-gold potential of the southern Arunta Region
- https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/74288 Southern Northern Territory uranium and geothermal energy systems assessment digitial data package
- https://www.researchgate.net/publication/342352173_Modelling_gold_potential_in_the_Granites-Tanami_Orogen_NT_Australia_A_comparative_study_using_continuous_and_data-driven_techniques
### NSW
- https://www.resourcesandgeoscience.nsw.gov.au/miners-and-explorers/geoscience-information/projects/mineral-potential-mapping#_southern-_new-_england-_orogen-mineral-potential
- https://search.geoscience.nsw.gov.au/product/9233 - 54 Curnamona Province and Delamerian-Thomson Orogen
- https://search.geoscience.nsw.gov.au/product/9253 - Eastern Lachlan Orogen https://www.smedg.org.au/GSNSW_2019_Blevin.pdf
- https://www.researchgate.net/publication/265915602_Comparing_prospectivity_modelling_results_and_past_exploration_data_A_case_study_of_porphyry_Cu-Au_mineral_systems_in_the_Macquarie_Arc_Lachlan_Fold_Belt_New_South_Wales
### England
- https://colab.research.google.com/drive/168PSo21-Jkwdz8xOmr5-rX9_DL3SInCN  [Tungsten Model Notebook]
 
### Brazil
- https://www.researchgate.net/publication/340633563_CATALOG_OF_PROSPECTIVITY_MAPS_OF_SELECTED_AREAS_FROM_BRAZIL
- https://www.researchgate.net/publication/340633739_MINERAL_POTENTIAL_AND_OPORTUNITIES_FOR_THE_EXPLORATION_OF_NEW_GEOLOGICAL_GROUNDS_IN_BRAZIL
- https://www.semanticscholar.org/paper/Mineral-Potential-Mapping-for-Orogenic-Gold-in-the-Silva-Silva/a23a9ce4da48863da876758afa9e1d2723088853
- https://www.scielo.br/scielo.php?script=sci_arttext&pid=S2317-48892016000200261 - Supergene nickel deposits in outhwestern Sao Francisco Carton, Brazil
#### Carajas
- https://www.researchgate.net/publication/258466504_Self-Organizing_Maps_A_Data_Mining_Tool_for_the_Analysis_of_Airborne_Geophysical_Data_Collected_over_the_Brazilian_Amazon
- https://www.researchgate.net/publication/258647519_Semiautomated_geologic_mapping_using_self-organizing_maps_and_airborne_geophysics_in_the_Brazilian_Amazon
- https://www.researchgate.net/publication/235443304_GIS-Based_prospectivity_mapping_for_orogenic_gold_A_case_study_from_the_Andorinhas_region_Brasil
- https://www.researchgate.net/publication/341936771_Modeling_of_Cu-Au_Prospectivity_in_the_Carajas_mineral_province_Brazil_through_Machine_Learning_Dealing_with_Imbalanced_Training_Data
- https://www.researchgate.net/publication/332031621_Predictive_lithological_mapping_through_machine_learning_methods_a_case_study_in_the_Cinzento_Lineament_Carajas_Province_Brazil
- https://www.researchgate.net/publication/340633659_Copper-gold_favorability_in_the_Cinzento_Shear_Zone_Carajas_Mineral_Province
- https://www.researchgate.net/publication/329477409_Favorability_potential_for_IOCG_type_deposits_in_the_Riacho_do_Pontal_Belt_New_insights_for_identifying_prospects_of_IOCG-type_deposits_in_NE_Brazil
- https://www.researchgate.net/publication/339453836_Uranium_anomalies_detection_through_Random_Forest_regression
- https://d1wqtxts1xzle7.cloudfront.net/48145419/Artificial_neural_networks_applied_to_mi20160818-5365-odv4na.pdf?1471522188=&response-content-disposition=inline%3B+filename%3DArtificial_neural_networks_applied_to_mi.pdf&Expires=1593477539&Signature=DNmSxKogrD54dE4LX~8DT4K7vV0ZGcf8Q2RRfXEPsCc8PGiBrbeBpy4NVQdCiENLz-YfSzVGk6LI8k5MEGxR~qwnUn9ISLHDuIau6VqBFSEA29jMixCbvQM6hbkUJKQlli-AuSPUV23TsSk76kB6amDYtwNHmBnUPzTQGZLj2XkzJza9PA-7W2-VrPQKHNPxJp3z8J0mPq4rhmHZLaFMMSL6QMpK5qpvSqi6Znx-kIhCprlyYfODisq0unOIwnEQstiMf2RnB6gPmGOodhNlLsSr01e7TvtvFDBOQvhhooeDeQrvkINN4DJjAIIrbrcQ8B2b-ATQS0a3QQe93h-VFA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA - Leite, E.P.L.; de Souza Filho, C.R. Artificial neural networks applied to mineral potential mapping for copper-gold mineralizations in the Carajás Mineral Province, Brazil. Geoph. Prosp. 2009, 57, 1049–1065.
- https://link-springer-com.access.library.unisa.edu.au/content/pdf/10.1007/s11053-015-9263-2.pdf - A Comparative Analysis of Weights of Evidence, Evidential Belief Functions, and Fuzzy Logic for Mineral Potential Mapping Using Incomplete Data at the Scale of Investigation
- https://library.seg.org/doi/abs/10.1190/sbgf2011-245 - Gold Prospectivity Mapping of Andorinhas Greenstone Belt, Para
#### Gurupi
- https://www.researchgate.net/publication/312220651_Predictive_Mapping_of_Prospectivity_in_the_Gurupi_Orogenic_Gold_Belt_North-Northeast_Brazil_An_Example_of_District-Scale_Mineral_System_Approach_to_Exploration_Targeting
  
### Australia
- https://www.researchgate.net/publication/260107484_Unsupervised_clustering_of_continental-scale_geophysical_and_geochemical_data_using_Self-Organising_Maps
- https://www.researchgate.net/publication/332263305_A_speedy_update_on_machine_learning_applied_to_bedrock_mapping_using_geochemistry_or_geophysics_examples_from_the_Pacific_Rim_and_nearby
- https://www.researchgate.net/publication/317312520_Catchment-based_gold_prospectivity_analysis_combining_geochemical_geophysical_and_geological_data_across_northern_Australia
- https://www.researchgate.net/publication/326571155_Continental-scale_mineral_prospectivity_assessment_using_the_National_Geochemical_Survey_of_Australia_NGSA_dataset
- https://www.researchgate.net/publication/334440382_Mapping_iron_oxide_Cu-Au_IOCG_mineral_potential_in_Australia_using_a_knowledge-driven_mineral_systems-based_approach
- https://www.researchgate.net/publication/282189370_Uranium_Prospectivity_Mapping_Across_the_Australian_Continent_via_Unsupervised_Cluster_Analysis_of_Integrated_Remote_Sensing_Data
#### South Australia
- https://www.researchgate.net/publication/335313790_Prospectivity_modelling_of_the_Olympic_Cu-Au_Province - https://services.sarig.sa.gov.au/raster/ProspectivityModelling/wms?service=wms&version=1.1.1&REQUEST=GetCapabilities
#### Queensland
- https://www.researchgate.net/publication/317312520_Catchment-based_gold_prospectivity_analysis_combining_geochemical_geophysical_and_geological_data_across_northern_Australia
- https://www.researchgate.net/publication/222211452_Predictive_modelling_of_prospectivity_for_Pb-Zn_deposits_in_the_Lawn_Hill_Region_Queensland_Australia
- https://www.researchgate.net/publication/252707107_GIS-based_epithermal_copper_prospectivity_mapping_of_the_Mt_Isa_Inlier_Australia_Implications_for_exploration_targeting
#### New South Wales
- https://www.researchgate.net/publication/337569823_Practical_Implementation_of_Random_Forest-Based_Mineral_Potential_Mapping_for_Porphyry_Cu-Au_Mineralization_in_the_Eastern_Lachlan_Orogen_NSW_Australia
- https://www.researchgate.net/publication/333551776_Translating_expressions_of_intrusion-related_mineral_systems_into_mappable_spatial_proxies_for_mineral_potential_mapping_Case_studies_from_the_Southern_New_England_Orogen_Australia
- https://www.researchgate.net/publication/336349643_MINERAL_POTENTIAL_MAPPING_AS_A_STRATEGIC_PLANNING_TOOL_IN_THE_EASTERN_LACHLAN_OROGEN_NSW
- https://www.researchgate.net/publication/329761040_NSW_Zone_54_Mineral_Systems_Mineral_Potential_Report
- https://www.publish.csiro.au/ex/pdf/ASEG2013ab236 - Mineral prospectivity analysis of the Wagga–Omeo belt in NSW
#### Tasmania
- https://www.researchgate.net/publication/262380025_Mapping_geology_and_volcanic-hosted_massive_sulfide_alteration_in_the_Hellyer-Mt_Charter_region_Tasmania_using_Random_Forests_TM_and_Self-Organising_Maps
- https://colab.research.google.com/drive/168PSo21-Jkwdz8xOmr5-rX9_DL3SInCN  [Tungsten Model Notebook]
#### Victoria
- https://www.researchgate.net/publication/323856713_Lithological_mapping_using_Random_Forests_applied_to_geophysical_and_remote_sensing_data_a_demonstration_study_from_the_Eastern_Goldfields_of_Australia
- https://publications.csiro.au/publications/#publication/PIcsiro:EP123339/SQmineral%20prospectivity/RP1/RS50/RORECENT/STsearch-by-keyword/LISEA/RI16/RT26 [nickel]
- https://www.researchgate.net/publication/257026553_Regional_prospectivity_analysis_for_hydrothermal-remobilised_nickel_mineral_systems_in_western_Victoria_Australia
####Western Australia
- https://www.researchgate.net/publication/274714146_Reducing_subjectivity_in_multi-commodity_mineral_prospectivity_analyses_Modelling_the_west_Kimberley_Australia
- https://www.researchgate.net/publication/319013132_Identifying_mineral_prospectivity_using_3D_magnetotelluric_potential_field_and_geological_data_in_the_east_Kimberley_Australia
- https://www.researchgate.net/publication/280930127_Regional-scale_targeting_for_gold_in_the_Yilgarn_Craton_Part_1_of_the_Yilgarn_Gold_Exploration_Targeting_Atlas
- https://www.researchgate.net/publication/279533541_District-scale_targeting_for_gold_in_the_Yilgarn_Craton_Part_2_of_the_Yilgarn_Gold_Exploration_Targeting_Atlas
- https://www.researchgate.net/publication/257026568_Exploration_targeting_for_orogenic_gold_deposits_in_the_Granites-Tanami_Orogen_Mineral_system_analysis_targeting_model_and_prospectivity_analysis
- https://www.researchgate.net/publication/280039091_Mineral_systems_approach_applied_to_GIS-based_2D-prospectivity_modelling_of_geological_regions_Insights_from_Western_Australia (the West Arunta Orogen, West Musgrave Orogen and Gascoyne Province - http://dmpbookshop.eruditetechnologies.com.au/product/mineral-systems-analysis-of-the-west-musgrave-province-regional-structure-and-prospectivity-modelling.do
- https://reader.elsevier.com/reader/sd/pii/S0169136810000417? - token=9FD1C06A25E7ECC0C384C0ECF976E4BC9C36047C53CEED08066811979A640E89DD94C49510D1B500C6FF5E69982E018E Prospectivity analysis of the Plutonic Marymia Greenstone Belt, Western Australia
- https://research-repository.uwa.edu.au/en/publications/exploration-targeting-for-orogenic-gold-deposits-in-the-granites- - Tanami orogen
- https://www.researchgate.net/publication/332631130_Fuzzy_inference_systems_for_prospectivity_modeling_of_mineral_systems_and_a_case-study_for_prospectivity_mapping_of_surficial_Uranium_in_Yeelirrie_Area_Western_Australia_Ore_Geology_Reviews_71_839-852Tasmania
- https://publications.csiro.au/rpr/download?pid=csiro:EP102133&dsid=DS3 [nickel]
- 
### Sweden
- https://www.researchgate.net/publication/259128115_Biogeochemical_expression_of_rare_earth_element_and_zirconium_mineralization_at_Norra_Karr_Southern_Sweden
- https://www.researchgate.net/publication/336086368_GIS-based_mineral_system_approach_for_prospectivity_mapping_of_iron-oxide_apatite-bearing_mineralisation_in_Bergslagen_Sweden
- https://www.researchgate.net/publication/260086862_COMPARISION_OF_VMS_PROSPECTIVITY_MAPPING_BY_EBF_AND_WOFE_MODELING_THE_SKELLEFTE_DISTRICT_SWEDEN
- https://www.researchgate.net/publication/229347041_Predictive_mapping_of_prospectivity_and_quantitative_estimation_of_undiscovered_VMS_deposits_in_Skellefte_district_Sweden
- https://www.researchgate.net/publication/260086947_PRELIMINARY_GIS-BASED_ANALYSIS_OF_REGIONAL-SCALE_VMS_PROSPECTIVITY_IN_THE_SKELLEFTE_REGION_SWEDEN
### Finland
- https://www.researchgate.net/publication/332298116_Scalability_of_the_Mineral_Prospectivity_Modelling_-_An_orogenic_gold_case_study_from_northern_Finland
- https://www.researchgate.net/publication/331006924_Unsupervised_clustering_and_empirical_fuzzy_memberships_for_mineral_prospectivity_modelling
- https://www.researchgate.net/publication/324517415_Can_boosting_boost_minimal_invasive_exploration_targeting
- https://www.researchgate.net/publication/320703774_Prospectivity_Models_for_Volcanogenic_Massive_Sulfide_Deposits_VMS_in_Northern_Finland
- https://www.researchgate.net/publication/320709733_Knowledge-driven_prospectivity_model_for_Iron_oxide-Cu-Au_IOCG_deposits_in_northern_Finland
- https://www.researchgate.net/publication/315381587_Introduction_to_the_special_issue_GIS-based_mineral_potential_targeting
- https://www.researchgate.net/publication/312180531_Optimizing_a_Knowledge-driven_Prospectivity_Model_for_Gold_Deposits_Within_Perapohja_Belt_Northern_Finland
- https://www.researchgate.net/publication/280875727_Receiver_operating_characteristics_ROC_as_validation_tool_for_prospectivity_models_-_A_magmatic_Ni-Cu_case_study_from_the_Central_Lapland_Greenstone_Belt_Northern_Finland
- https://www.researchgate.net/publication/283451958_Data-driven_logistic-based_weighting_of_geochemical_and_geological_evidence_layers_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/320280611_Evaluation_of_boosting_algorithms_for_prospectivity_mapping
- https://www.researchgate.net/publication/298297988_Fuzzy_logic_data_integration_technique_used_as_a_nickel_exploration_tool
- https://www.researchgate.net/publication/259372191_Gravity_data_in_regional_scale_3D_and_gold_prospectivity_modelling_-_example_from_the_Central_Lapland_greenstone_belt_northern_Finland
- https://www.researchgate.net/publication/251786465_Spatial_data_analysis_as_a_tool_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/248955109_Combined_conceptualempirical_prospectivity_mapping_for_orogenic_gold_in_the_northern_Fennoscandian_Shield_Finland
- https://www.researchgate.net/publication/332352805_Boosting_for_Mineral_Prospectivity_Modeling_A_New_GIS_Toolbox
- https://publications.csiro.au/publications/#publication/PIcsiro:EP146125/SQmineral%20prospectivity/RP1/RS50/RORECENT/STsearch-by-keyword/LISEA/RI12/RT26
  
### Norway
- https://www.mdpi.com/2075-163X/9/2/131/htm - Prospectivity Mapping of Mineral Deposits in Northern Norway Using Radial Basis Function Neural Networks
 
### Spain
- https://www.researchgate.net/publication/222198648_Knowledge-guided_data-driven_evidential_belief_modeling_of_mineral_prospectivity_in_Cabo_de_Gata_SE_Spain
- https://www.researchgate.net/publication/263542579_Optimal_Exploration_Target_Zones
- https://www.researchgate.net/publication/225656353_Deriving_Optimal_Exploration_Target_Zones_on_Mineral_Prospectivity_Maps
- https://www.researchgate.net/publication/43165602_Methodology_for_deriving_optimal_exploration_target_zones
- https://www.researchgate.net/publication/222892103_Optimal_field_sampling_for_targeting_minerals_using_hyperspectral_data
 Peru
### England
- https://www.researchgate.net/publication/342339753_A_machine_learning_approach_to_tungsten_prospectivity_modelling_using_knowledge-driven_feature_extraction_and_model_confidence
- https://www.researchgate.net/project/Enhancing-the-Geological-Understanding-of-SW-England-Using-Machine-Learning-Algorithms
### Africa
- https://www.researchgate.net/publication/340084035_Reliability_of_using_ASTER_data_in_lithologic_mapping_and_alteration_mineral_detection_of_the_basement_complex_of_West_Berenice_Southeastern_Desert_Egypt
- https://www.researchgate.net/publication/260792212_Nickel_Sulphide_Deposits_in_Archaean_Greenstone_Belts_in_Zimbabwe_Review_and_Prospectivity_Analysis
### Uganda
- https://www.researchgate.net/publication/262566098_Predictive_Mapping_of_Prospectivity_for_Orogenic_Gold_in_Uganda
- https://www.researchgate.net/publication/242339962_Predictive_mapping_for_orogenic_gold_prospectivity_in_Uganda
### Ghana
- https://www.researchgate.net/publication/227256267_Application_of_Data-Driven_Evidential_Belief_Functions_to_Prospectivity_Mapping_for_Aquamarine-Bearing_Pegmatites_Lundazi_District_Zambia
- https://www.researchgate.net/publication/226842511_Mapping_of_prospectivity_and_estimation_of_number_of_undiscovered_prospects_for_lode_gold_southwestern_Ashanti_Belt_Ghana
- https://www.researchgate.net/publication/233791624_Spatial_association_of_gold_deposits_with_remotely_-_sensed_faults_South_Ashanti_belt_Ghana
### Zambia
- https://www.researchgate.net/publication/263542565_APPLICATION_OF_REMOTE_SENSING_AND_SPATIAL_DATA_INTEGRATION_TO_PREDICT_POTENTIAL_ZONES_FOR_AQUAMARINE-BEARING_PEGMATITES_LUNDAZI_AREA_NORTHEAST_ZAMBIA
- https://www.researchgate.net/publication/264041472_Geological_and_Mineral_Potential_Mapping_by_Geoscience_Data_Integration
### Central Africa
- https://www.researchgate.net/publication/323452014_The_Utility_of_Machine_Learning_in_Identification_of_Key_Geophysical_and_Geochemical_Datasets_A_Case_Study_in_Lithological_Mapping_in_the_Central_African_Copper_Belt
- https://www.researchgate.net/publication/334436808_Lithological_Mapping_in_the_Central_African_Copper_Belt_using_Random_Forests_and_Clustering_Strategies_for_Optimised_Results
### South Africa
- https://www.researchgate.net/publication/264296137_PREDICTIVE_BEDROCK_AND_MINERAL_PROSPECTIVITY_MAPPING_IN_THE_GIYANI_GREENSTONE_BELT_SOUTH_AFRICA
- https://www.researchgate.net/publication/268196204_Predictive_mapping_of_prospectivity_for_orogenic_gold_Giyani_greenstone_belt_South_Africa
### China
- https://www.researchgate.net/publication/329037175_Mineral_prospectivity_analysis_for_BIF_iron_deposits_A_case_study_in_the_Anshan-Benxi_area_Liaoning_province_North-East_China
- https://www.researchgate.net/publication/229399579_Mapping_geochemical_singularity_using_multifractal_analysis_Application_to_anomaly_definition_on_stream_sediments_data_from_Funin_Sheet_Yunnan_China
- https://www.researchgate.net/publication/338871759_Modeling-based_mineral_system_approach_to_prospectivity_mapping_of_stratabound_hydrothermal_deposits_A_case_study_of_MVT_Pb-Zn_deposits_in_the_Huayuan_area_northwestern_Hunan_Province_China
- https://www.researchgate.net/publication/235443301_Mineral_potential_mapping_in_a_frontier_region
- https://www.researchgate.net/publication/235443302_Mineral_potential_mapping_in_frontier_regions_A_Mongolian_case_study
- https://www.researchgate.net/publication/332751556_Application_of_hierarchical_clustering_singularity_mapping_and_Kohonen_neural_network_to_identify_Ag-Au-Pb-Zn_polymetallic_mineralization_associated_geochemical_anomaly_in_Pangxidong_district
- https://www.researchgate.net/publication/332547136_Prospectivity_Mapping_for_Porphyry_Cu-Mo_Mineralization_in_the_Eastern_Tianshan_Xinjiang_Northwestern_China
- https://www.researchgate.net/publication/338789096_From_2D_to_3D_Modeling_of_Mineral_Prospectivity_Using_Multi-source_Geoscience_Datasets_Wulong_Gold_District_China
- https://www.researchgate.net/publication/336771580_3D_Mineral_Prospectivity_Mapping_with_Random_Forests_A_Case_Study_of_Tongling_Anhui_China
- https://www.researchgate.net/publication/334106787_Mapping_Mineral_Prospectivity_via_Semi-supervised_Random_Forest
- https://www.researchgate.net/publication/331575655_Mapping_Geochemical_Anomalies_Through_Integrating_Random_Forest_and_Metric_Learning_Methods
- https://www.researchgate.net/publication/340401748_Effects_of_Random_Negative_Training_Samples_on_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/329600793_A_combined_approach_using_spatially-weighted_principal_components_analysis_and_wavelet_transformation_for_geochemical_anomaly_mapping_in_the_Dashui_ore-concentration_district_Central_China
- https://www.researchgate.net/publication/329299202_Integrating_sequential_indicator_simulation_and_singularity_analysis_to_analyze_uncertainty_of_geochemical_anomaly_for_exploration_targeting_of_tungsten_polymetallic_mineralization_Nanling_belt_South_
- https://www.researchgate.net/publication/328623280_Maximum_Entropy_and_Random_Forest_Modeling_of_Mineral_Potential_Analysis_of_Gold_Prospectivity_in_the_Hezuo-Meiwu_District_West_Qinling_Orogen_China
- https://www.researchgate.net/publication/328255422_Mapping_mineral_prospectivity_through_big_data_analytics_and_a_deep_learning_algorithm
- https://www.researchgate.net/publication/325702993_Assessment_of_Geochemical_Anomaly_Uncertainty_Through_Geostatistical_Simulation_and_Singularity_Analysis
- https://www.researchgate.net/publication/267927676_Evaluation_of_uncertainty_in_mineral_prospectivity_mapping_due_to_missing_evidence_A_case_study_with_skarn-type_Fe_deposits_in_Southwestern_Fujian_Province_China
- https://www.researchgate.net/publication/267927506_GIS-based_mineral_potential_modeling_by_advanced_spatial_analytical_methods_in_the_southeastern_Yunnan_mineral_district_China
- https://www.researchgate.net/publication/307011381_Identification_and_mapping_of_geochemical_patterns_and_their_significance_for_regional_metallogeny_in_the_southern_Sanjiang_China
- https://www.researchgate.net/publication/236270466_Mapping_of_district-scale_potential_targets_using_fractal_models
- https://www.researchgate.net/publication/339096362_Application_of_nonconventional_mineral_exploration_techniques_case_studies
### USA
- https://www.researchgate.net/publication/338663292_A_Predictive_Geospatial_Exploration_Model_for_Mississippi_Valley_Type_Pb-Zn_Mineralization_in_the_Southeast_Missouri_Lead_District
-  
-  
### Canada
- https://www.researchgate.net/publication/220164155_Support_vector_machine_A_tool_for_mapping_mineral_prospectivity
- https://www.researchgate.net/publication/343511849_Identification_of_intrusive_lithologies_in_volcanic_terrains_in_British_Columbia_by_machine_learning_using_Random_Forests_the_value_of_using_a_soft_classifier
### Iran
- https://www.researchgate.net/publication/336471932_A_knowledge-guided_fuzzy_inference_approach_for_integrating_geophysics_geochemistry_and_geology_data_in_deposit-scale_porphyry_copper_targeting_Saveh-Iran
- https://www.researchgate.net/publication/330129457_Performance_evaluation_of_RBF-_and_SVM-based_machine_learning_algorithms_for_predictive_mineral_prospectivity_modeling_integration_of_S-A_multifractal_model_and_mineralization_controls
- https://www.researchgate.net/publication/320886789_Prospectivity_analysis_of_orogenic_gold_deposits_in_Saqez-Sardasht_Goldfield_Zagros_Orogen_Iran
- https://www.researchgate.net/publication/258505300_Application_of_staged_factor_analysis_and_logistic_function_to_create_a_fuzzy_stream_sediment_geochemical_evidence_layer_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/270586282_Data-Driven_Index_Overlay_and_Boolean_Logic_Mineral_Prospectivity_Modeling_in_Greenfields_Exploration
- https://www.researchgate.net/publication/296638839_An_AHP-TOPSIS_Predictive_Model_for_District-Scale_Mapping_of_Porphyry_Cu-Au_Potential_A_Case_Study_from_Salafchegan_Area_Central_Iran
- https://www.researchgate.net/publication/310658663_Multifractal_interpolation_and_spectrum-area_fractal_modeling_of_stream_sediment_geochemical_data_Implications_for_mapping_exploration_targets
- https://www.researchgate.net/publication/220164381_Application_of_geochemical_zonality_coefficients_in_mineral_prospectivity_mapping
- https://research-repository.uwa.edu.au/en/publications/exploration-feature-selection-applied-to-hybrid-data-integration-Exploration feature selection applied to hybrid data integrationmodeling: Targeting copper-gold potential in central 
### Iran
#### NW
- https://www.researchgate.net/publication/278029106_Application_of_Discriminant_Analysis_and_Support_Vector_Machine_in_Mapping_Gold_Potential_Areas_for_Further_Drilling_in_the_Sari-Gunay_Gold_Deposit_NW_Iran
- https://www.researchgate.net/publication/317240761_Enhancement_and_Mapping_of_Weak_Multivariate_Stream_Sediment_Geochemical_Anomalies_in_Ahar_Area_NW_Iran
- https://www.researchgate.net/publication/267635150_Multivariate_regression_analysis_of_lithogeochemical_data_to_model_subsurface_mineralization_A_case_study_from_the_Sari_Gunay_epithermal_gold_deposit_NW_Iran
- https://www.researchgate.net/publication/339153591_Sensitivity_analysis_of_prospectivity_modeling_to_evidence_maps_enhancing_success_of_targeting_for_epithermal_gold_Takab_district_NW_Iran
- https://www.researchgate.net/publication/304904242_Stepwise_regression_for_recognition_of_geochemical_anomalies_Case_study_in_Takab_area_NW_Iran
- 
- 
### Argentina
- https://www.researchgate.net/publication/277940917_Porphyry_epithermal_and_orogenic_gold_prospectivity_of_Argentina
- https://www.researchgate.net/publication/235443303_Prospectivity_mapping_for_multi-stage_epithermal_gold_mineralization_in_Argentina
- https://www.researchgate.net/publication/269518805_Prospectivity_for_epithermal_gold-silver_deposits_in_the_Deseado_Massif_Argentina
- https://www.researchgate.net/publication/263542560_EVIDENTIAL_BELIEF_MAPPING_OF_EPITHERMAL_GOLD_POTENTIAL_IN_THE_DESEADO_MASSIF_SANTA_CRUZ_PROVINCE_ARGENTINA
- https://www.researchgate.net/publication/263542691_ANALYSIS_OF_SPATIAL_DISTRIBUTION_OF_EPITHERMAL_GOLD_DEPOSITS_IN_THE_DESEADO_MASSIF_SANTA_CRUZ_PROVINCE
### Chile
- https://www.researchgate.net/publication/341485750_Evaluation_of_random_forest-based_analysis_for_the_gypsum_distribution_in_the_Atacama_desert
### Phillipines
- https://www.researchgate.net/publication/267927677_Data-driven_predictive_mapping_of_gold_prospectivity_Baguio_district_Philippines_Application_of_Random_Forests_algorithm
- https://www.researchgate.net/publication/267640864_Random_forest_predictive_modeling_of_mineral_prospectivity_with_small_number_of_prospects_and_data_with_missing_values_in_Abra_Philippines
- https://www.researchgate.net/publication/276271833_Data-Driven_Predictive_Modeling_of_Mineral_Prospectivity_Using_Random_Forests_A_Case_Study_in_Catanduanes_Island_Philippines
- https://www.researchgate.net/publication/229641286_Improved_Wildcat_Modelling_of_Mineral_Prospectivity
- https://www.researchgate.net/publication/263174923_Application_of_Mineral_Exploration_Models_and_GIS_to_Generate_Mineral_Potential_Maps_as_Input_for_Optimum_Land-Use_Planning_in_the_Philippines
- https://www.researchgate.net/publication/241001432_Geologically_Constrained_Probabilistic_Mapping_of_Gold_Potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/263724277_Geologically_Constrained_Fuzzy_Mapping_of_Gold_Mineralization_Potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/3931975_Remote_detection_of_vegetation_stress_for_mineral_exploration
- https://www.researchgate.net/publication/238447208_Logistic_Regression_for_Geologically_Constrained_Mapping_of_Gold_Potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/263422015_Where_Are_Porphyry_Copper_Deposits_Spatially_Localized_A_Case_Study_in_Benguet_Province_Philippines
- https://www.researchgate.net/publication/233488614_Wildcat_mapping_of_gold_potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/248977334_Mineral_imaging_with_Landsat_TM_data_for_hydrothermal_alteration_mapping_in_heavily-vegetated_terrane​​​​​​
- https://www.researchgate.net/publication/272092198_Logistic_Regression_for_Geologically_Constrained_Mapping_of_Gold_Potential_Baguio_District
- https://www.researchgate.net/publication/209803275_Evidential_belief_functions_for_data-driven_geologically_constrained_mapping_of_gold_potential_Baguio_district_Philippines
- https://www.researchgate.net/publication/209803275_Evidential_belief_functions_for_data-driven_geologically_constrained_mapping_of_gold_potential_Baguio_district_Philippines
- https://www.researchgate.net/publication/226982180_Weights_of_Evidence_Modeling_of_Mineral_Potential_A_Case_Study_Using_Small_Number_of_Prospects_Abra_Philippines
India
- https://www.researchgate.net/publication/272092276_Extended_Weights-of-Evidence_Modelling_for_Predictive_Mapping_of_Base_Metal_Deposit_Potential_in_Aravalli_Province_Western_India
- https://www.researchgate.net/publication/226193283_Knowledge-Driven_and_Data-Driven_Fuzzy_Models_for_Predictive_Mineral_Potential_Mapping
- https://www.researchgate.net/publication/227221497_Artificial_Neural_Networks_for_Mineral-Potential_Mapping_A_Case_Study_from_Aravalli_Province_Western_India
- https://www.researchgate.net/publication/226092981_A_Hybrid_Neuro-Fuzzy_Model_for_Mineral_Potential_Mapping
- https://www.researchgate.net/publication/222050039_Bayesian_network_classifiers_for_mineral_potential_mapping
- https://www.researchgate.net/publication/225328359_A_Hybrid_Fuzzy_Weights-of-Evidence_Model_for_Mineral_Potential_Mapping
- https://www.researchgate.net/publication/238027981_SVM-based_base-metal_prospectivity_modeling_of_the_Aravalli_Orogen_Northwestern_India
### Indonesia
- https://www.researchgate.net/publication/263542819_Regional-Scale_Geothermal_Prospectivity_Mapping_in_West_Java_Indonesia_by_Data-driven_Evidential_Belief_Functions
 
## Endowment Modelling
- https://www.researchgate.net/publication/342405763_Predicting_grade-tonnage_characteristics_of_undiscovered_mineralisation_application_of_the_USGS_Three-part_Undiscovered_Mineral_Resource_Assessment_to_the_Sandstone_Greenstone_Belt_of_the_Yilgarn_Bloc
- https://github.com/iagoslc/ZipfsLaw_Quadrilatero_Ferrifero
- https://www.researchgate.net/publication/341087909_Assessing_the_variability_of_expert_estimates_in_the_USGS_Three-part_Mineral_Resource_Assessment_Methodology_A_call_for_increased_skill_diversity_and_scenario-based_training
- https://www.researchgate.net/publication/308778798_Spatial_analysis_of_mineral_deposit_distribution_A_review_of_methods_and_implications_for_structural_controls_on_iron_oxide-copper-gold_mineralization_in_Carajas_Brazil
- https://www.researchgate.net/publication/222834436_Controls_on_mineral_deposit_occurrence_inferred_from_analysis_of_their_spatial_pattern_and_spatial_association_with_geological_features
- https://www.researchgate.net/publication/229347041_Predictive_mapping_of_prospectivity_and_quantitative_estimation_of_undiscovered_VMS_deposits_in_Skellefte_district_Sweden
- https://www.researchgate.net/publication/229792860_From_Predictive_Mapping_of_Mineral_Prospectivity_to_Quantitative_Estimation_of_Number_of_Undiscovered_Prospects
- https://www.researchgate.net/publication/238365283_Metal_endowment_of_cratons_terranes_and_districts_Insights_from_a_quantitative_analysis_of_regions_with_giant_and_super-giant_deposits
- https://www.sciencedirect.com/science/article/pii/S0169136810000685
- https://www.researchgate.net/publication/330994502_Global_Grade-and-Tonnage_Modeling_of_Uranium_deposits
- https://www.researchgate.net/publication/240301743_Spatial_statistical_analysis_of_the_distribution_of_komatiite-hosted_nickel_sulfide_deposits_in_the_Kalgoorlie_terrane_Western_Australia_Clustered_or_Not
- https://www.researchgate.net/publication/248211962_A_new_method_for_spatial_centrographic_analysis_of_mineral_deposit_clusters
- https://pubs.geoscienceworld.org/segweb/economicgeology/article-abstract/103/4/829/127993/Linking-Mineral-Deposit-Models-to-Quantitative?redirectedFrom=fulltext
- https://www.researchgate.net/publication/275620329_A_Time-Series_Audit_of_Zipf's_Law_as_a_Measure_of_Terrane_Endowment_and_Maturity_in_Mineral_Exploration
## World Models
- https://www.researchgate.net/publication/325344128_The_role_of_basement_control_in_Iron_Oxide-Copper-Gold_mineral_systems_revealed_by_satellite_gravity_models
- https://www.researchgate.net/publication/331283650_Archean_crust_and_metallogenic_zones_in_the_Amazonian_Craton_sensed_by_satellite_gravity_data
- https://www.researchgate.net/publication/331428028_Supplementary_Material_for_the_paper_Archean_crust_and_metallogenic_zones_in_the_Amazonian_Craton_sensed_by_satellite_gravity_data
- https://www.researchgate.net/post/Is_it_possible_to_derive_free_air_anomaly_or_bouguer_anomaly_from_gravity_disturbance_data
- https://www.leouieda.com/pdf/use-the-disturbance.pdf
- https://www.leouieda.com/papers/use-the-disturbance.html
- https://eartharxiv.org/2kjvc/ -> Global distribution of sediment-hosted metals controlled by craton edge stability
 
## Financial Forecasting
- https://www.researchgate.net/publication/317137060_Forecasting_copper_prices_by_decision_tree_learning
- https://www.researchgate.net/publication/4874824_Mine_Size_and_the_Structure_of_Costs

## Agent based Modelling
- https://mpra.ub.uni-muenchen.de/62159/ -> Mineral exploration as a game of chance [Agent Based Modelling]
 
 
