# mineral-exploration-machine-learning
This page lists resources for mineral exploration and machine learning, generally with useful code and examples. 
ML and Data Science is a huge field, these are resources I have found useful and/or interesting to me in practice.
Links currently to a fork of a repository are because I have changed something to use and put in a list for reference.
Resources are also given for data analysis, transformation and visualisation as that is most of the work.

Suggestions welcome: open a discussion, issue or pull request.

# Table of Contents

* [Prospectivity](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#prospectivity)
* [Geology](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#geology)
* [Natural Language Processing](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#natural-language-processing)
* [Remote Sensing](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#remote-sensing)
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
* [Datasets](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#datasets)
* [Papers](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#papers)
* [Other](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#other)
* [General Interest](https://github.com/RichardScottOZ/mineral-exploration-machine-learning#general-interest)

# Map
* [Map of Github](https://anvaka.github.io/map-of-github/#10.26/28.263/19.7499)
* [DeepWiki](https://github.com/AsyncFuncAI/deepwiki-open) -> automagic wiki style analysis of a repo via llm

# Frameworks
* [UNCOVER-ML Framework](https://github.com/RichardScottOZ/uncover-ml)
  * [Geo-Wavelets](https://github.com/RichardScottOZ/geo-wavelets)
  * [ML-Preprocessing](https://github.com/GeoscienceAustralia/ML-preprocessing)
  * [GIS ML Workflow](https://github.com/sheecegardezi/GIS-ML-Workflow)
* [EIS Toolkit](https://github.com/GispoCoding/eis_toolkit/tree/master) -> Python library for mineral prospectivity mapping from EIS Horizon EU Project
* [PySpatialML](https://github.com/RichardScottOZ/Pyspatialml) -> Library that facilitates prediction and handling for raster machine learning automatically to geotiff, etc.
* [DARPA Criticalmaas](https://github.com/DARPA-CRITICALMAAS)
  * [competition info](file:///C:/Users/rscott/Downloads/DARPA-PA-22-02-01.pdf)
* [scikit-map](https://github.com/scikit-map/scikit-map/tree/master)
* [TorchGeo](https://github.com/microsoft/torchgeo) -> Pytorch library for remote sensing style models
 * [terratorch](https://github.com/IBM/terratorch) -> Flexible fine-tuning framework for Geospatial Foundation Models
  * [TorchSpatial](https://github.com/seai-lab/TorchSpatial)
	* [paper](https://arxiv.org/abs/2406.15658)
* [torch-harmonics](https://github.com/NVIDIA/torch-harmonics) -> Differentiable signal processing on the sphere for PyTorch
* [geodl](https://eartharxiv.org/repository/view/7417/)  
* [Geo Deep Learning](https://github.com/RichardScottOZ/geo-deep-learning) -> Simple deep learning framework based on RGB
* [AIDE: Artificial Intelligence for Disentangling Extremes](https://github.com/RichardScottOZ/AIDE?tab=readme-ov-file)
	* [paper](https://www.researchgate.net/profile/Miguel-Angel-Fernandez-Torres/publication/381917888_The_AIDE_Toolbox_Artificial_intelligence_for_disentangling_extreme_events/links/66846648714e0b03153f38ae/The-AIDE-Toolbox-Artificial-intelligence-for-disentangling-extreme-events.pdf)
* [ExPloRA](https://github.com/samar-khanna/ExPLoRA) -> ExPLoRA: Parameter-Efficient Extended Pre-training to Adapt Vision Transformers under Domain Shifts
	* [Website](https://www.samarkhanna.com/ExPLoRA/)
	* [paper](https://arxiv.org/abs/2406.10973)
 
* [pyClusterwise](https://pypi.org/project/pyClusterWise/)
  * [paper] -> https://www.sciencedirect.com/science/article/pii/S0169136825001519?via%3Dihub -> Clustering in geo-data science: Navigating uncertainty to select the most reliable method
* [GeoStat Framework](https://github.com/GeoStat-Framework) -> Group of repositories with kriging and other

## R
* [CAST](https://github.com/RichardScottOZ/CAST) -> Caret Applications for Spatio-Temporal models
* [geodl](https://github.com/maxwell-geospatial/geodl) -> semantic segmentation of geospatial data using convolutional neural network-based deep learning
	* [paper](https://arxiv.org/html/2404.06978v1)
### Pipelines
* [geotargts](https://github.com/RichardScottOZ/geotargets) -> Extension of targets to terra and stars


# Prospectivity
## Oceania
### Australia
* [Iron oxide copper-gold mineral potential maps](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/149222)
* [Lateritic Ni-Co prospectivity modeling in eastern Australia using an enhanced generative adversarial network and positive-unlabeled bagging] -> (https://zenodo.org/records/14037494)
  * [paper](https://link.springer.com/article/10.1007/s11053-024-10423-4)
* [Machine learning for geological mapping : algorithms and applications](https://eprints.utas.edu.au/18571/) -> PhD thesis with code and data
* [minpot-toolkit](https://github.com/GeoscienceAustralia/minpot-toolkit/tree/main) -> Example of Hoggard et al Lab Boundary analysis with Sedimentary copper
* [MPM-WofE](https://github.com/GeoscienceAustralia/MPM-WofE) -> Mineral Potential Mapping - Weights of Evidence
* [Porphyry Copper Spatio-Temporal Exploration](https://github.com/EarthByte/porphyry_copper_spatiotemporal_exploration)  
	* [paper](https://www.earthbyte.org/predicting-the-emplacement-of-cordilleran-porphyry-copper-systems-using-a-spatio-temporal-machine-learning-model/)
* [Prospectivity Mapping of Ni-Co Laterites](https://github.com/EarthByte/Lachlan_Laterite_Ni_Co)
* [Transform 2022 Tutorial](https://github.com/Solve-Geosolutions/transform_2022) -> Random forest example
	* [Video](https://www.youtube.com/watch?v=C4YvnLMzYDc)
* [Tin-Tungsten](https://medium.com/@thomas.ostersen/tin-tungsten-prospecting-with-machine-learning-in-northeast-tasmania-australia-3c23519f81cf)
	* [Collab](https://colab.research.google.com/drive/168PSo21-Jkwdz8xOmr5-rX9_DL3SInCN?usp=sharing)
#### Explorer Challenge
* [Explorer Challenge](https://github.com/RichardScottOZ/explore_australia) -> OZ Minerals run competition with Data Science introduction
#### South Australia
* [Gawler_MPM](https://github.com/e-farahbakhsh/Gawler_MPM) -> Cobalt, Chromium, Nickel
	* [Paper](https://www.researchgate.net/publication/373954003_Critical_mineral_prospectivity_mapping_on_the_Gawler_craton_using_a_new_machine_learning_framework)
* [Geophysical Data Clustering in the Gawler Craton](https://github.com/EarthByte/geophysical_image_clustering_exploration)
	* [Zenodo Data](Automated detection of mineralization-related craton structures using geophysical data and unsupervised machine learning)
#### Explore SA - South Australian Department of Energy and Mining Competition
* [Winners](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/abba4f54-b6ef-4fe4-b951-57f11299d490) -> SARIG data information
* [Caldera](https://github.com/mrodda117/CalderaPublic) -> Caldera Analytics analysis
* [IncertoData](https://github.com/RichardScottOZ/ExploreSA/tree/master/Data_submission_competition)
* [Butterworth and Barnett](https://github.com/RichardScottOZ/gawler-exploration) -> Butterworth and Barnett entry
* [Data Driven Mineralisation Mapping](https://github.com/Abdallah-M-Ali/mman4020_Data_Driven_Mineralisation_Mapping/blob/main/docs/tensor.ipynb)

### Papua New Guinea
* [SpatioTemporAl Mineral Prospectivity (STAMP) Modelling](https://github.com/EarthByte/STAMP_PNG)
	* [Zenodo Data](https://zenodo.org/records/14189755)

## North America
### Canada
* [Transfer Prospectivity Learnnig](https://github.com/Anagabrielamantilla/TransferProspectivityLearning)
	* [paper](https://www.sciencedirect.com/science/article/pii/S1342937X24002727)  -> Porphyry-type mineral prospectivity mapping with imbalanced data via prior geological transfer learning

## South America
* [Machine learning to classify ore deposits from tectonomagmatic properties](https://github.com/intelligent-exploration/mineralexplorationcourse/tree/master/Week10)
## Brazil
* [Mapa Preditivo](https://github.com/fnaghetini/Mapa-Preditivo) -> Brazil student project
* [Course_Predictive_Mapping_USP](https://github.com/victsnet/Curso_Mapeamento_Preditivo_USP) -> Course Project
* [Mineral Prospectivity Mapping](https://github.com/Eliasmgprado/MineralProspectivityMapping)
* [3D Weights of Evidence](https://github.com/e-farahbakhsh/3DWofE)
* [Geological Complexity SMOTE](https://github.com/Eliasmgprado/GeologicalComplexity_SMOTE) -> includes fractal analysis
	* [paper](https://doi.org/10.1016/j.oregeorev.2020.103611)
* [MPM Jurena](https://github.com/victsnet/MPM---Juruena-Mineral-Province ) -> Jurena Mineral Province
## China
* [MPM by ensemble learning](https://github.com/ZhiqiangZhangCUGB/MPM-by-ensemble-learning) -> Qingchengzi Pb-Zn-Ag-Au polymetallic district China
* [Mineral Prospectivity Prediction Convolutional Neural Networks](https://github.com/yangna815/Mineral-Prospectivity-Prediction-Convolutional-Neural-Networks) -> CNN Example with a few architectures [a paper by this author uses GoogleNet]
* [Mineral Prospectivity Prediction by CSAE](https://github.com/yangna815/Mineral-Prospectivity-Prediction-by-CSAE)
* [Mineral Prospectivity Prediction by CAE](https://github.com/yangna815/Mineral-Prospectivity-Prediction-by-CAE)
 	* [paper](https://www.researchgate.net/publication/350817877_Mineral_Prospectivity_Prediction_via_Convolutional_Neural_Networks_Based_on_Geological_Big_Data)
## Sudan
* [Mineral Prospectivity Mapping ML](https://github.com/Abdallah-M-Ali/Mineral-Prospectivity-Mapping-ML)
## Norway
* [A machine learning–based approach to regional-scale mapping of sensitive glaciomarine clay combining airborne electromagnetics and geotechnical data](https://github.com/emerald-geomodelling/publication-NSG2021-Christensen-QuickClayML)
	* [paper](https://onlinelibrary.wiley.com/doi/10.1002/nsg.12166)
## World
* [Porphyry prospectivity global](https://github.com/EarthByte/porphyry-prospectivity-global) -> Spatio-temporal global mapping
  * [zenodo](https://zenodo.org/record/8157691)

# Geology
* [Brazil Predictive Geology Maps](https://github.com/marcosbr/predictive-geology-maps) -> Work by the Brazil geological survey
* [depth to bedrock](https://github.com/Alberta-Geological-Survey/depth-to-bedrock)(Evaluating spatially enabled machine learning approaches for depth to bedrock mapping) 
	* [paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0296881)
* [DL-RMD](https://github.com/rizwanasif/DL-RMD) -> A geophysically constrained electromagnetic resistivity model database for deep learning applications
* [Earthscape](https://github.com/masseygeo/earthscape) -> EarthScape: A Multimodal Dataset for Surficial Geologic Mapping and Earth Surface Analysis
  * [paper](https://arxiv.org/abs/2503.15625)
* [Geological Image Classifier](https://github.com/PCleverleyGeol/Geological-Image-Classifier)
* [Geological mapping in the age of artificial intelligence](https://geoscientist.online/sections/features/geological-mapping-in-the-age-of-artificial-intelligence/) -> Geological mapping in the age of artificial intelligence
* [GeolNR](https://github.com/MichaelHillier/GeoINR) -> Geological Implicit Neural Representation for three-dimensional structural geological modelling applications
	* [paper](https://gmd.copernicus.org/articles/16/6987/2023/)
* [mapeamento_litologico_preditivo](https://github.com/Gabriel-Goes/mapeamento_litologico_preditivo)
* [Mapping Global Lithospheric Mantle Pressure-Temperature Conditions by Machine-Learning Thermobarometry](https://zenodo.org/records/8353966)
	* [paper](https://www.researchgate.net/publication/379639953_Mapping_Global_Lithospheric_Mantle_Pressure-Temperature_Conditions_by_Machine-Learning_Thermobarometry)
* [Neural Rock Typing](https://github.com/LukasMosser/neural_rock_typing)
* [geological map explanatory power](https://github.com/charliekirkwood/geological-map-explanatory-power)
  * [paper](https://www.lyellcollection.org/doi/full/10.1144/esss2024-005) -> Quantifying the limited explanatory power of geological maps
* [PEACE](https://github.com/microsoft/PEACE) -> Empowering Geologic Map Holistic Understanding with MLLMs)
  *[paper](https://github.com/microsoft/PEACE) -> GeoMap-Bench
* [SimCLR Core Disturbance](simclr_core_disturbance)
  * [paper](https://www.sciencedirect.com/science/article/pii/S2590197424000636?via%3Dihub)
* [West Musgraves Geology Uncertainty](https://medium.com/@thomas.ostersen/uncertainty-mapping-in-the-west-musgraves-australia-988fc49ce1e4) -> Uncertainty map prediction with entropy analysis: highly useful
* [Non Stationarity Mitigation Transformer](https://github.com/LeiLiu-cloud/NonstationarityMitigation_Transformer)
  * [Collab](https://colab.research.google.com/drive/1pPZCSjlNPn7_n8GLGCxY0bo7QmbVU65G?usp=sharing) -> Notebook
	* [paper](https://www.sciencedirect.com/science/article/abs/pii/S0098300423001164)
* [Bedrock-vs-sediment](https://github.com/alexandra-jarna/Bedrock-vs-sediment)
	* [paper](https://www.researchgate.net/publication/370929945_Where_are_the_outcrops_Automatic_delineation_of_bedrock_from_sediments_using_Deep-Learning_techniques
* [autoencoders_remotesensing](https://github.com/sydney-machine-learning/autoencoders_remotesensing)
	* [paper](https://arxiv.org/abs/2404.02180v1) -> Remote sensing framework for geological mapping via stacked autoencoders and clustering
* [Geological Section Generated based on per-pixel linked lists](https://github.com/Dy111111/Geological-Sections_Generation_based_on_PPLL )
  * [paper](https://www.researchgate.net/publication/391580813_A_real-time_geological_sections_generation_method_for_geological_3D_models_based_on_per-pixel_linked_lists)

## Training Data
* [Into the Noddyverse](https://github.com/Loop3D/noddyverse/tree/1.0) -> a massive data store of 3D geological models for machine learning and inversion applications
  * [Original Zenondo repository](https://zenodo.org/record/4589883#.YvYk23ZByUk)  
  * [website](https://tectonique.net/noddy)
  * [New Geonetwork Location](https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f3570_5134_6366_9231)
* [Geo-Bench](https://huggingface.co/datasets/microsoft/PEACE)

## Lithology
* [Deep Learning Lithology](https://github.com/RichardScottOZ/deeplearning_lithology)
* [Rock Protolith Predictor](https://github.com/RichardScottOZ/Rock_protolith_predictor)
* [SA Geology Lithology Predictions](https://github.com/RADutchie/SA-geology-litho-predictions)
* [Automated Well Log Correlation](https://github.com/dudley-fitzgerald/AutomatedWellLogCorrelation)
* [dawson-facies-2022](https://github.com/johnlab-research/dawson-facies-2022) -> Transfer learning for geological images
	* [paper](https://www.sciencedirect.com/science/article/pii/S0098300422002333) - > Impact of dataset size and convolutional neural network architecture on transfer learning for carbonate rock classification
* [Litho-Classification](https://github.com/luthfigeo/Litho-Classification) -> Volcanic facies Classification using Random Forest
* [Multi-view ensemble machine learning approach for 3D modeling using geological and geophysical data](https://doi.org/10.6084/m9.figshare.23647590.v1)
* [Petromind](https://github.com/connamulder/PetroMind) -> PetroMind: A multimodal petrographic model for rock image classification and lithological description generation
  * [paper](https://www.sciencedirect.com/science/article/pii/S2590197425000965)
	* [paper](https://www.tandfonline.com/doi/abs/10.1080/13658816.2024.2394228)
* [SedNet](https://github.com/MudRocw1/SedNet_explainable-deep-learning-network)
	* [paper](https://www.sciencedirect.com/science/article/abs/pii/S0098300423002157?via%3Dihub)
* [SCB-Net](https://github.com/victsnet/SCB-Net) -> Lithological mapping using Spatially Constrained Bayesian Network
  * [paper](https://www.sciencedirect.com/science/article/pii/S0098300425001141?via%3Dihub)

## Drilling
* [Heterogenous Drilling](https://geoscienceaustralia.github.io/uncover-ml/projectreport.html#uncoverml-project-report-nicta) - Nicta/Data61 project report for looking at modelling using drillholes that don't go far enough
* [corel](https://github.com/RichardScottOZ/corel) -> smart computer vision model that identifies facies and performs rock typing on core images

## Paleovalleys
* [Sub3DNet1.0: a deep-learning model for regional-scale 3D subsurface structure mapping](https://gmd.copernicus.org/articles/14/3421/2021/)

## Stratigraphy
* [Predicatops](https://github.com/JustinGOSSES/predictatops) -> Stratigraphic predication designed for hydrocarbon
* [stratal-geometries](https://github.com/jessepisel/stratal-geometries) -> Predicting Stratigraphic Geometries from Subsurface Well Logs

## Structural
* [APGS](https://github.com/ondrolexa/apsg) -> Structural geology package
* [Assessing plate reconstruction models using plate driving force consistency tests](https://zenodo.org/records/7904975) -> Jupyter notebook and data
	* [paper](https://www.nature.com/articles/s41598-023-37117-w)
* [gplately](https://github.com/GPlates/gplately)
* [structural geology cookbook](https://github.com/gcmatos/structural-geology-cookbook]
* [GEOMAPLEARN 1.0](https://doi.org/10.18144/8aee-7b77) -> Detecting geological structures from geological maps with machine learning
* [Lineament Learning](https://github.com/aminrd/LineamentLearning) -> Fault prediction and mapping via potential field deep learning and clustering
* [LitMod3D](https://github.com/javfurchu/LitMod3D_V3.1) -> 3D integrated geophysical-petrological interactive modelling of the lithosphere and underlying upper mantle
* [Machine Learning Based Mohometry](https://github.com/EarthByte/Machine-Learning-Based-Mohometry/tree/main) -> Paleo Crustal Thickness Estimation and Evolution Visualization
  * [paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JB030404)
* [others](https://www.bgh.org.au/software/)
* [StructuralGeo](https://github.com/eldadHaber/StructuralGeo)
  * [zenodo](https://zenodo.org/records/15244035)
  * [paper](https://arxiv.org/abs/2506.11164) -> Synthetic Geology: Structural Geology Meets Deep Learning


## Simulation
* [GebPy](https://github.com/MABeeskow/GebPy) -> generation of geological data for rocks and minerals
* [OpenGeoSys](https://gitlab.opengeosys.org/ogs/ogs) -> development of numerical methods for the simulation of thermo-hydro-mechanical-chemical (THMC) processes in porous and fractured media
* [Stratigraphics.jl](https://github.com/JuliaEarth/StratiGraphics.jl) -> Creating 3D stratigraphy from 2D geostatistical processes

## Geodynamics
* [Badlands](https://github.com/badlands-model/badlands) -> Basin and Landscape Dynamics
* [CitcomS](https://github.com/geodynamics/citcoms) -> finite element code designed to solve compressible thermochemical convection problems relevant to Earth's mantle.
* [LaMEM](https://github.com/UniMainzGeo/LaMEM) -> simulate various thermo-mechanical geodynamical processes such as mantle-lithosphere interaction
* [PTatin3D](https://bitbucket.org/ptatin/ptatin3d/src/master/) -> studying long time-scale processes relevant to geodynamics [original motivation :toolkit capable of studying high-resolution, three-dimensional models of lithospheric deformation]
* [underworld](https://github.com/underworldcode/underworld2) -> Finite element modelling of geodynamics
* [Discern detachment of the subducting slab in an ancient subduction zone using machine learning](https://zenodo.org/records/14059244)
  * [paper](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024JH000400)

## Geophysics
### General
* [Feature fusion-enhanced t-SNE image atlas for geophysical features discovery](https://github.com/ll-portes/t-sne_atlas)
  * [paper](https://www.nature.com/articles/s41598-025-01333-3)

### Structural
* [Structural Geophysics Tools](https://github.com/swaxi/SGTool) -> QGIS aimed
### Foundation Models
* [Cross-Domain Foundation Model Adaptation: Pioneering Computer Vision Models for Geophysical Data Analysis](https://github.com/programmerzxg/cross-domain-foundation-model-adaptation) -> some of code to come
	* [paper](https://arxiv.org/pdf/2408.12396)
* [Seismic Foundation Model](https://github.com/shenghanlin/SeismicFoundationModel) -> "a new generation deep learning model in geophysics"
* [Geological Everything Model] (https://arxiv.org/abs/2507.00419) -> Geological Everything Model 3D: A Physics-informed Promptable Foundation Model for Unified and Zero-Shot Subsurface Understanding

### Australia
#### Regolith Depth
* [Regolith Depth](https://data.csiro.au/collection/csiro:11393v6) -> Model
* [Complete Radiometrics Grid of Australia with modelled infill](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/144413)
#### AEM Interpolation
* [High resolution conductivity mapping using regional AEM survey](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146163)
	* [Abstract](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146380)
  
### Electromagnetics
* [TEM-NLnet: A Deep Denoising Network for Transient Electromagnetic Signal with Noise Learning](https://github.com/wmyCDUT/TEM-NLnet_demo)

### Inversion
* [Machine Learning and Geophysical Inversion](https://github.com/ezygeo-ai/machine-learning-and-geophysical-inversion) -> reconstruct paper from Y. Kim and N. Nakata (The Leading Edge, Volume 37, Issue 12, Dec 2018)

#### Euler deconvolution
- https://legacy.fatiando.org/gallery/gravmag/euler_moving_window.html
	- Harmonica version eventually? https://hackmd.io/@fatiando/development-calls-2024?utm_source=preview-mode&utm_medium=rec
- https://notebook.community/joferkington/tutorials/1404_Euler_deconvolution/euler-deconvolution-examples
- https://github.com/ffigura/Euler-deconvolution-plateau


### Gravity
* [Recovering 3D Basement Relief Using Gravity Data Through Convolutional Neural Networks]
	* [Data](10.5281/zenodo.5543969)
	* [paper](https://www.researchgate.net/publication/355118950_Recovering_3D_Basement_Relief_Using_Gravity_Data_Through_Convolutional_Neural_Networks)
* [Stable downward continuation of the gravity potential field implemented using deep learning](https://github.com/LiYongbo-geo/DC-Net-code)
	* [paper](https://www.researchgate.net/publication/366965954_Stable_downward_continuation_of_the_gravity_potential_field_implemented_using_deep_learning)
* [Fast imaging for the 3D density structures by machine learning approach](https://github.com/LiYongbo-geo/GV-Net-code)
	* [paper](https://www.researchgate.net/publication/366922016_Fast_imaging_for_the_3D_density_structures_by_machine_learning_approach)
 
### Magnetics
* [High-resolution aeromagnetic map through Adapted-SRGAN](https://github.com/MBS1984/Adapted-SRGAN)
	* [paper](https://www.sciencedirect.com/science/article/pii/S0098300423000675)
* [MagImage2Geo3D](https://github.com/neu-gjt/MagImage2Geo3D)
	* [paper](https://www.researchgate.net/publication/348697645_3D_geological_structure_inversion_from_Noddy-generated_magnetic_data_using_deep_learning_methods)

### Seismic
* [StorSeismic](https://github.com/swag-kaust/storseismic) -> An approach to pre-train a neural network to store seismic data features
* [PINNtomo](https://github.com/tianyining/PINNtomo) -> Seismic tomography using physics-informed neural networks
	* [paper](https://arxiv.org/abs/2104.01588)
### Seismology
* [obspy](https://github.com/obspy/obspy) -> framework for processing seismological

### Petrophysics
* [ML4Rocks](https://github.com/clberube/ml4rocks) -> Some intro work

### Tectonics
* [Discern detachment of the subducting slab in an ancient subduction zone using machine learning](https://github.com/dzheng2333/basalt_geochemistry) -> Notebook
	* [figshare](https://figshare.com/articles/dataset/Supporting_information_for_A_machine_learning_approach_to_identify_the_abrupt_transition_of_tectonic_settings_using_trace_elemental_dataset_of_basalts_/24015024)
* [Colab notebook](https://data.csiro.au/collection/csiro:61119) -> Google Colab input file for benchmark results of ML-SEISMIC publication
	* [paper](https://www.researchgate.net/publication/376892064_Physics-informed_neural_network_reconciles_Australian_displacements_and_tectonic_stresses)
* [Unleashing the power of Machine
Learning in Geodynamics](https://github.com/GiteonCaulfied/COMP4560_stokes_ml_project])
	* [Honours Thesis](https://www.researchgate.net/profile/Xuzeng-He/publication/380316113_Unleashing_the_power_of_Machine_Learning_in_Geodynamics/links/6634eac535243041535c878b/Unleashing-the-power-of-Machine-Learning-in-Geodynamics.pdf)
  [related](https://github.com/GiteonCaulfied/COMP4560_stokes_ml_project/blob/main/misc_pdfs/Atkins_Thesis_Finding_the_patterns_in_mantle_convection.pdf)
* [Physics-Infomred Neural Networks for fault slip simulation with rate and state friction law](https://github.com/RichardScottOZ/PINN_3DSSE/tree/main)
	* [simulation and frictional paramter estimation on slow slip events](https://zenodo.org/records/13731480)
	* [paper](https://d197for5662m48.cloudfront.net/documents/publicationstatus/223843/preprint_pdf/29c1d7b3b99de49d81f41725303268db.pdf) -> Physics-Informed Deep Learning for Estimating the Spatial Distribution of Frictional Parameters in Slow Slip Regions
	
  
# Geochemistry
* [CODAinPractice](https://github.com/michaelgreenacre/CODAinPractice) -> Compositional Data Analysis in Practice
* [GeoCoDa](https://www.researchgate.net/publication/372487589_GeoCoDA_Recognizing_and_Validating_Structural_Processes_in_Geochemical_Data_A_Workflow_on_Compositional_Data_Analysis_in_Lithogeochemistry)
* [DAN-GRF](https://github.com/Saeid1986/DAN-GRF) -> Deep autoencoder network connected to geographical random forest for spatially aware geochemical anomaly detection
	* [paper](https://www.sciencedirect.com/science/article/pii/S0098300424001407)
* [Dash Geochemical Prospection](https://github.com/pvabreu7/DashGeochemicalProspection) -> Web-app classifying stream sediments with K-means
* [Enhancing machine learning thermobarometry for clinopyroxene-bearing magmas](https://github.com/magredal/Enhancing-ML-Thermobarometry-for-Clinopyroxene-Bearing-Magmas)
	* [paper](https://www.sciencedirect.com/science/article/pii/S0098300424001900) -> Enhancing-ML-Thermobarometry-for-Clinopyroxene-Bearing-Magmas
* [Zircon fertility models](https://github.com/cicarrascog/Zircon_fertility_models) -> Decision trees to predict fertile zircon from porphyry copper deposits
	* [paper](https://www.researchgate.net/publication/382089226_Quantifying_the_Criteria_Used_to_Identify_Zircons_from_Ore-Bearing_and_Barren_Systems_in_Porphyry_Copper_Exploration)
* [Machine Learning Zircon Trace Element Tool to Predict Porphyry Deposit Type and Resource Size](https://github.com/ZihaoWen123/PDT_PCR_PMR/tree/v0.01	)
	* [zenodo archive](https://zenodo.org/records/10292176)
	* [paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2024JH000180)
* [geology_class0](https://github.com/ZihaoWen123/geology_class) -> A machine learning approach to discrimination of igneous rocks and ore deposits by zircon trace elements 
	* [paper](https://www.degruyter.com/document/doi/10.2138/am-2022-8899/html)
	* [Demo application](http://60.205.170.161:8001/)
	* https://colab.research.google.com/drive/1-bOZgG6Nxt2Rp1ueO1SYmzIqCRiyyYcT
* [GeochemPrint](https://colab.research.google.com/drive/1nX0vip0VS3f-GjL9l5femBiRcMwMGe_Y) 
	* [paper](https://www.researchgate.net/publication/363754140_Geochemical_fingerprinting_of_continental_and_oceanic_basalts_A_machine_learning_approach)
* [Global geochemistry](https://github.com/dhasterok/global_geochemistry)
* [ICBMS Jacobina](https://github.com/gferrsilva/icpms-jacobina) -> Analysis of pyrite chemistry from a gold deposit
* [Interpretation of Trace Element Chemistry of Zircons from Bor and Cukaru Peki: Conventional Approach and Random Forest Classification](https://www.mdpi.com/2076-3263/12/11/396)
* [indicator_minerals](https://github.com/DinaKlim/indicator_minerals) -> Can PCA can tell the story of the origin of tourmaline?
* [Journal of Geochemical Exploration - Manifold](https://github.com/geometatqueens/2020---Journal-of-Geochemical-Exploration--Manifold)
* [LewisML](https://github.com/RichardScottOZ/LewisML) -> Analysis of the Lewis Formation
* [MICA](https://github.com/bluetyson/MICA_shiny) -> Chemical composition, in Shiny
* [Multivariate statistical analysis and bespoke deviation network modeling for geochemical anomaly detection of rare earth elements](https://github.com/EarthByte/MPM_Geochemistry_Curnamona_REE/tree/main)
	* [paper](https://www.sciencedirect.com/science/article/pii/S0883292724002518?dgcid=coauthor)
* [Prospectivity mapping of rare earth elements through geochemical data analysis](https://github.com/EarthByte/MPM_Geochemistry_Curnamona_REE) -> Prospectivity mapping of rare earth elements through geochemical data analysis
* [QMineral Modeller](https://github.com/gferrsilva/QMineral_Modeller) -> Mineral Chemistry virtual assistant from the Brazilian geological survey
* [Secular Changes in the Occurrence of Subduction During the Archean](https://zenodo.org/records/10418615) -> Zenodo code archive
	* [paper] https://www.researchgate.net/publication/380289934_Secular_Changes_in_the_Occurrence_of_Subduction_During_the_ArcheanA machine learning approach to discrimination of igneous rocks and ore deposits by zircon trace elements 
* [geochemical anomaly detection](https://github.com/MinersAI/geochemical_anomaly_detection) -> Multivariate Outlier Detection in Geochemical Datasets
 
# Kriging
* [DKNN: deep kriging neural network for interpretable geospatial interpolation](https://github.com/in1311/DKNN)
	* [paper](https://www.tandfonline.com/doi/full/10.1080/13658816.2024.2347316 )

# Natural Language Processing
## Frameworks
* [spacy](https://github.com/explosion/spaCy) -> NLP Library
## Text Extraction
* [Text Extraction](https://github.com/RichardScottOZ/amazon-textract-textractor) -> Text extraction from documents : paid ML as a service, but works very well, can extract tables efficiently
	* [Large Scale](https://github.com/RichardScottOZ/amazon-textract-serverless-large-scale-document-processing) -> Large scale version
* [NASA Concept Tagging](https://github.com/RichardScottOZ/concept-tagging-training) -> Keyword prediction 
	* [API](https://github.com/nasa/concept-tagging-api) -> API web service
	* [Presentation](https://datascience.jpl.nasa.gov/aiworkshop/presentation-27)
* [Petrography Report Data Extractor](https://github.com/RichardScottOZ/Petrography-report-data-extractor)
* [SA Exploration Topic Modelling](https://github.com/RADutchie/SA-exploration-topic-modelling) -> Topic modelling from exploration reports
* [Stratigraph](https://github.com/BritishGeologicalSurvey/stratigraph)
* [Geocorpus](https://github.com/jneto04/geocorpus)
* [Portuguese BERT](https://github.com/neuralmind-ai/portuguese-bert)
* [BERT CWS](https://github.com/cugdeeplearn/BERTCWS)
	* [paper](https://www.researchgate.net/publication/352009328_Chinese_Word_Segmentation_Based_on_Self-Learning_Model_and_Geological_Knowledge_for_the_Geoscience_Domain)
* [Automated Extraction of Mining Company Drillhole Results](https://github.com/RichardScottOZ/Automatic-Extraction-of-Mining-Company-Drillhole-Results)
	* [Conference Paper](https://aclanthology.org/2022.wnut-1.16/)

### PDF handling
* [pdfminer](https://github.com/pdfminer/pdfminer.six)
* [pdfplumber](https://github.com/jsvine/pdfplumber) -> pdf table extraction
* [pikepdf](https://github.com/RichardScottOZ/pikepdf) -> pdf image extraction
* [PyMuPDF](https://github.com/pymupdf/PyMuPDF) -> pdf parser
#### Tables
* [camelot](https://github.com/camelot-dev/camelot) -> pdf text table extraction
* [layoutparser](https://github.com/Layout-Parser/layout-parser) -> deep learning layhout detection
* [messytables](https://github.com/okfn/messytables?tab=readme-ov-file) -> find headers and datatypes

## Heuristic Aids
### Java
* [GeoTopicParser](https://cwiki.apache.org/confluence/display/TIKA/GeoTopicParser)
### Python
* [GIS Metadata parsing](https://github.com/RichardScottOZ/gis-metadata-parser) -> extract data from xml etc.
### R
* [crssuggest](https://github.com/walkerke/crsuggest) -> coordinate reference system suggerstions
* [tidyxl](https://github.com/cran/tidyxl)
### OCR
* [Apache Tika](https://tika.apache.org/) -> OCR, content analysis
* [Parsee PDF Reader](https://github.com/parsee-ai/parsee-pdf-reader) - PDF Reading/OCR
* [Tesseract](https://github.com/tesseract-ocr/tesseract) -> OCR
## Word Embeddings
* [Geoscience Language Models](https://github.com/NRCan/geoscience_language_models) -> processing code pipeline and models [Glove, BERT) retrained on geoscience documents from Canada
	* [Datasets](https://geoscan.nrcan.gc.ca/starweb/geoscan/servlet.starweb?path=geoscan/downloade.web&search1=R=329265) -> Data to support models
	* [paper](https://doi.org/10.1016/j.acags.2022.100084) -> Geoscience language models and their intrinsic evaluation
	* [paper](https://link.springer.com/article/10.1007/s11053-023-10216-1) -> Applications of Natural Language Processing to Geoscience Text Data and Prospectivity Modeling
* [GeoVec](https://github.com/spadarian/GeoVec) -> Word embedding model trained on 300K geoscience papers
	* [GeoVec Model](https://osf.io/4uyeq/) -> OSF Storage for GeoVec model
   	* [paper](https://soil.copernicus.org/articles/5/177/2019/)
    * [GeoVecto Litho](https://github.com/IFuentesSR/GeoVectoLitho) -> 3D Models interpolation from word embeddings
	* [GeoVEC Playground](https://github.com/RichardScottOZ/geoVec-playground) -> Working with the Padarian GeoVec glove word embeddings model
* [GloVe](https://github.com/stanfordnlp/GloVe) -> Standford library for producing word embeddings
	* [gloVE python](https://pypi.org/project/glove-python-binary/) glove, glove-python highly problematic on windows: here Binary version for Windows installs:
* [Mittens](https://github.com/roamanalytics/mittens) -> In memory vectorized glove implementation 
* [PetroNLP](https://github.com/Petroles/PetroNLP) -> Organisation
  * [PetroVec](https://github.com/Petroles/Petrovec) -> Portuguese Word Embeddings for the Oil and Gas Industry: development and evaluation
  * [Petro KGraph](https://github.com/Petroles/Petro_KGraph) -> Ontological work with petrovec
  * [paper](https://www.sciencedirect.com/science/article/abs/pii/S0098300424001973) -> [UNSEEN]
* [wordembeddingsOG](https://github.com/diogosmg/wordEmbeddingsOG) -> Portuguese Oil and Gas word embeddings
* [Portuguese Word Embeddings](https://github.com/nathanshartmann/portuguese_word_embeddings) 
	* [Portuguese models](http://nilc.icmc.usp.br/embeddings)
* [Spanish Word Embeddings](https://github.com/dccuchile/spanish-word-embeddings)
* [Multilingual alignment](https://github.com/babylonhealth/fastText_multilingual/blob/master/align_your_own.ipynb)
	* [Overview of approaches](https://www.jair.org/index.php/jair/article/view/11640/26511)
* [Application-of-natural-language-processing-for-finding-semantic-relation-of-elusive-natural-resource](https://github.com/NanmanasLin/Application-of-natural-language-processing-for-finding-semantic-relation-of-elusive-natural-resource)
  * [paper](https://arxiv.org/abs/2504.07490) -> Geological Inference from Textual Data using Word Embeddings
## Named Entity Recognition
* [Geo NER Model](https://github.com/BritishGeologicalSurvey/geo-ner-model) -> Named entity recognition
* [GeoBERT](https://huggingface.co/botryan96/GeoBERT) - hugging face repo for model in  
	* [paper]https://www.researchgate.net/publication/359186219_Few-shot_learning_for_name_entity_recognition_in_geological_text_based_on_GeoBERT
* [GLiNER](https://github.com/urchade/GLiNER) -> Few shot deep learning NER
* [INDUS](https://huggingface.co/nasa-impact/nasa-smd-ibm-v0.1) -> NASA science tailored LLM suite
	* [paper](https://arxiv.org/html/2405.10725v2)
* [How to find key geoscience terms in text without mastering NLP using Amazon Comprehend](https://github.com/aws-samples/amazon-comprehend-energy-geoscience-nlp)
* [OzRock](https://github.com/majiga/OzROCK) - OzRock: A labeled dataset for entity recognition in geological (mineral exploration) domain
## Ontology
* [GAKG](https://github.com/geobrain-ai/gakg?tab=readme-ov-file) -> A Multimodal Geoscience Academic Knowledge Graph (Chinese)
* [GeoERE-Net](https://github.com/GISer-WB/GeoERE-Net) -> Understanding geological reports based on knowledge graphs using a deep learning approach
	* [paper](https://www.researchgate.net/publication/363408251_Understanding_geological_reports_based_on_knowledge_graphs_using_a_deep_learning_approach)
* [GeoFault Ontology](https://github.com/Yuanwei-Q/GeoFault-Ontology) 
* [geosim](https://github.com/smolang/SemanticObjects/tree/geosim) -> Semantically Triggered Qualitative Simulation of a Geological Process
	* [https://www.duo.uio.no/handle/10852/111467](Knowledge Modelling for Digital Geology) -> PhD thesis with two papers
	* [SIRIUS GeoAnnotator](http://158.37.63.37:8081/gic) -> Website example from above
* [Ontology CWS](https://github.com/cugdeeplearn/OntologyCWS)
* [Stratigraphic Knowledge Graph (StraKG)](https://github.com/IGCCP/StraKG)
	* [paper](https://www.sciencedirect.com/science/article/pii/S2590197424000119)
## Spelling
* [pyenchant](https://github.com/pyenchant/pyenchant) -> spelling checker

## Large Language Models
* [JiuZhou](https://github.com/THU-ESIS/JiuZhou) -> Open Foundation Language Models for Geoscience
  * [paper](https://www.tandfonline.com/doi/pdf/10.1080/17538947.2025.2449708)
* [Large Language Model for Geoscience](https://github.com/davendw49/k2)
	* [Learning Foundation Language Models for Geoscience Knowledge Understanding and Utilization paper](https://arxiv.org/pdf/2306.05064.pdf)
* [GeoGalactica](https://github.com/geobrain-ai/geogalactica) -> A Larger foundation language model in Geoscience
	* [technical report](https://arxiv.org/abs/2401.00434)
* [GeoChat](https://github.com/mbzuai-oryx/geochat) -> grounded Large Vision Language Model for Remote Sensing
* [GeoMinLM](https://github.com/wangcug/GeoMinLM) -> GeoMinLM: A Large Language Model in Geology and Mineral Survey in Yunnan Province
  * [paper](https://www.sciencedirect.com/science/article/pii/S0169136825001982)
* [LAGDAL](https://github.com/JustinGOSSES/LAGDAL) -> LLM Matching geology map information to location experiments
* [OmniGeo](https://arxiv.org/abs/2503.16326) -> Towards a Multimodal Large Language Models for Geospatial Artificial Intelligence
* [GeoAssist](https://github.com/PCleverleyGeol/GeoAssist---An-open-source-autonomous-research-agent-for-geoscience-data-and-literature) ->  Autonomous-research-agent-for-geoscience-data-and-literature

### Chatbots
* [GeoGPT](https://geogpt.deep-time.org/universal-login) -> Deep Time Digital Earth Research Group from China project
* [GeoGPT](https://geogpt.zero2x.org/) -> Deep Time Digital Earth Research Group from China project
* [GeoGPT-Research-Project](https://github.com/GeoGPT-Research-Project/GeoGPT)

### Agents
* [GeoAgent](https://github.com/Yusin2Chen/GeoAgent) -> An LLM Agent for Automatic Geospatial Data Analysis
  * [paper](https://arxiv.org/abs/2410.18792)

# Remote Sensing
* [CNN Sentinel](https://github.com/jensleitloff/CNN-Sentinel) -> Overview about land-use classification from satellite data with CNNs based on an open dataset
* [DEA notebooks](https://github.com/GeoscienceAustralia/dea-notebooks/tree/develop/Real_world_examples/Scalable_machine_learning) -> Scalable machine learning example but lots of useful things here
* [EASI cookbook notebooks](https://github.com/csiro-easi/easi-notebooks/) -> CSIRO Earth Analytics platform introductions for ODC style analysis
* [DS_UNet](https://github.com/SebastianHafner/DS_UNet) -> Unet fusing Sentinel-1 Synthetic Aperture Radar (SAR) and Sentinel-2 Multispectral Imager 
* [Multi Pretext Masked Autoencoder (MP-MAE)](https://github.com/vishalned/MMEarth-train)
 * [data](https://github.com/vishalned/MMEarth-data)
* [segment-geospatial](https://github.com/opengeos/segment-geospatial) -> Segment anything for geospatial uses
 * [SamGIS](https://github.com/trincadev/samgis-be) -> Segment Anything applied to GIS
* [SatMAE++](https://github.com/techmn/satmae_pp) -> Rethinking Transformers Pre-training for Multi-Spectral Satellite Imagery
* [grid-mae](https://github.com/RichardScottOZ/grid-mae) -> Investigate using multiscale grids in a Vision Transformer Masked Autoencoder
* [ScaleMae](https://github.com/bair-climate-initiative/scale-mae)
	* [paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Reed_Scale-MAE_A_Scale-Aware_Masked_Autoencoder_for_Multiscale_Geospatial_Representation_Learning_ICCV_2023_paper.pdf)
* [CIMAE](https://github.com/Modexus/torchgeo) -> CIMAE - Channel Independent Masked Autoencoder
 * [fork](https://github.com/RichardScottOZ/torchgeo-cimae) -> to give it the name for reference
 * [Self-Supervised Representation Learning for Remote Sensing] -> Master's thesis includes the above and comparisons of several models
* [U Barn](https://src.koda.cnrs.fr/iris.dumeur/ssl_ubarn)
	* [paper](https://www.researchgate.net/publication/377712228_Self-Supervised_Spatio-Temporal_Representation_Learning_Of_Satellite_Image_Time_Series) 
* [earthnets](https://earthnets.nicepage.io/ )
* [GeoTorchAI](https://github.com/wherobots/GeoTorchAI) -> GeoTorchAI: A Spatiotemporal Deep Learning Framework
* [pytorcheo](https://github.com/earthpulse/pytorchEO -> Deep Learning for Earth Observation applications and research
* [pytorch cloud geotiff optimization](https://github.com/microsoft/pytorch-cloud-geotiff-optimization)
  * [paper](https://arxiv.org/pdf/2506.06235) -> Optimizing Cloud-to-GPU Throughput for Deep Learning With Earth
Observation Data
* [AiTLAS](https://github.com/biasvariancelabs/aitlas-arena) -> an open-source benchmark suite for evaluating state-of-the-art deep learning approaches for image classification in Earth Observation
* [Segmentation Gym](https://github.com/Doodleverse/segmentation_gym) ->  Gym is designed to be a "one stop shop" for image segmentation on "N-D" - any number of coincident bands in a multispectral image
* [deep_learning_alteration_zones](https://github.com/sydney-machine-learning/deeplearning_alteration_zones)
* [awesome mining band ratio collection](https://github.com/rodreras/awesome-mining-band-ratio) -> collection of simple band ratio uses for highlight various minerals
## Foundation Models
* [awesome remote sensing foundation models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)
  * [ChatEarthNet](https://github.com/zhu-xlab/ChatEarthNet)
  * [Zenodo](https://zenodo.org/records/11004358)
    * [paper] -> a global-scale image–text dataset empowering vision–language geo-foundation models
 * [Clay](https://github.com/Clay-foundation/model) -> An open source AI model and interface for Earth
 * [GeoDINO]A Vision Foundation Model for Earth Observation Leveraging DINO Architecture and Sentinel-2 Multi-Spectral Data
 * [IBM-NASA-GEOSPATIAL Prithvi](https://huggingface.co/ibm-nasa-geospatial)
  * [Image segmentation by foundation model finetuning](https://github.com/NASA-IMPACT/hls-foundation-os) -> For Prithvi
 * [AM-RADIO: Agglomerative Vision Foundation Model](https://github.com/NVlabs/RADIO)
 	* [paper](https://arxiv.org/abs/2312.06709) ->  - Reduce All Domains Into One
 * [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP) -> A Vision Language Foundation Model for Remote Sensing
 * [SpectralGPT](https://github.com/danfenghong/IEEE_TPAMI_SpectralGPT)
  * [zenodo](https://zenodo.org/records/8412455))  -> remote sensing foundation model customized for spectral data
 	* [paper](https://ieeexplore.ieee.org/document/10490262) -> Unseen
* [Terramind]  (https://github.com/IBM/terramind) -> any-to-any generative foundation model for Earth Observation
  * [paper](https://arxiv.org/abs/2504.11171) -> Terramind paper

  
 ## Processing
 * [ASTER Conversion](https://git.earthdata.nasa.gov/projects/LPDUR/repos/aster-l1t/browse) -> Conversion from ASTER hd5 to geotiff NASA github
 * [HLS Data Resources](https://github.com/nasa/HLS-Data-Resources) -> Harmonized Landsat Sentinel wrangling
 * [sarsen](https://github.com/bopen/sarsen) -> xarray based SAR image processing and correction
 * [openEO](https://github.com/Open-EO) -> openEO develops an open API to connect R, Python, JavaScript and other clients to EO cloud back-ends
 
## Spectral Unmixing
* [Conventional-to-Transformer-for-Hyperspectral-Image-Classification-Survey-2024](https://github.com/mahmad00/Conventional-to-Transformer-for-Hyperspectral-Image-Classification-Survey-2024)
	* [paper](https://arxiv.org/abs/2404.14955)
* [Hyperspectral Deep Learning Review](https://github.com/mhaut/hyperspectral_deeplearning_review)
* [Hyperspectral Autoencoders](https://github.com/RichardScottOZ/hyperspectral-autoencoders)
* [Deeplearn HSI](https://github.com/hantek/deeplearn_hsi)
* [3DCAE-hyperspectral-classification](https://github.com/MeiShaohui/3DCAE-hyperspectral-classification)
* [DeHIC](https://github.com/jingge326/DeHIC)
* [Rev-Net](https://github.com/Lab-PANbin/Rev-Net)
	* [paper](https://ieeexplore.ieee.org/document/10536904) -> A Reversible Generative Network for Hyperspectral Unmixing With Spectral Variability
* [Pysptools](https://github.com/RichardScottOZ/pysptools) -> also has useful heuristic algorithms
* [Spectral Python](https://github.com/spectralpython/spectral)
* [Spectral Dataset RockSL](https://github.com/RichardScottOZ/spectral-dataset-RockSL) -> Open spectral dataset
* [Unmixing](https://github.com/RichardScottOZ/unmixing)
* [Unmamba](https://github.com/Preston-Dong/UNMamba) -> Cascaded Spatial–Spectral Mamba for Blind Hyperspectral Unmixing
  * [paper](https://ieeexplore.ieee.org/document/10902420)

A Joint Multi-Scale Graph Attention and Classify-Driven Autoencoder Framework for Hyperspectral Unmixing

## Hyperspectral
* [CasFormer: Cascaded Transformers for Fusion-aware Computational Hyperspectral Imaging](https://github.com/danfenghong/Information_Fusion_CasFormer)
* [Spectral Normalization for Keras](https://github.com/IShengFang/SpectralNormalizationKeras) 
	* [paper](https://arxiv.org/abs/1802.05957)
* [S^2HM^2](https://github.com/tulilin/S2HM2) -> S2HM2: A Spectral-Spatial Hierarchical Masked Modeling Framework for Self-Supervised Feature Learning and Classification of Large-Scale Hyperspectral Image
	* [paper](https://ieeexplore.ieee.org/abstract/document/10508226)


## Visualisation
* [Deep Colormap Extraction from Visualizations](https://github.com/yuanlinping/deep_colormap_extraction)
	* [paper](https://arxiv.org/pdf/2103.00741.pdf)
* [Semantic Segmentation for Extracting Historic Surface Mining Disturbance from Topographic Maps](https://github.com/maxwell-geospatial/topoDL) -> Example is for coal mines
	* [paper](https://www.mdpi.com/2072-4292/12/24/4145)
* [International Chronostratigraphic Color Codes](https://stratigraphy.org/chart) -> RGB codes and others in spreadsheet and other formats
* [LithClass](https://mrdata.usgs.gov/catalog/lithrgb.txt) -> USGS version of lithology color codes
 * [color version](https://mrdata.usgs.gov/catalog/lithclass-color.php)
* [SeisWiz](https://github.com/amustafa9/SeisWiz) -> Lightweight python SEG-Y viewer

## Texture
* [Mineral Texture Classification Using Deep Convolutional Neural Networks: An Application to Zircons From Porphyry Copper Deposits](https://github.com/ChetanNathwani/zirconCNN)
	* [paper](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2022JB025933)
  
## Simulation
* [Intelligent Prospector](https://github.com/sisl/MineralExploration) -> Sequential data acquisition planning
 * [Zenodo](https://zenodo.org/records/6727378)
	* [paper](https://gmd.copernicus.org/articles/16/289/2023/gmd-16-289-2023.html)
	
## Geometry
* [Deep Angle](https://github.com/ArashRabbani/DeepAngle) -> Fast calculation of contact angles in tomography images using deep learning	

## Other
* [Network Analysis of Mineralogical Systems](https://github.com/lic10/DTDI-DataAnalysis)
  * [Data](http://www.minsocam.org/MSA/AmMin/TOC/2017/Aug2017_data/AM-17-86104.zip) -> Data from paper here
* [Geoanalytics and machine learning](https://github.com/victsnet/Geoanalytics-and-Machine-Learning)
* [Machine Learning Subsurface](https://github.com/PyBrown/Machine-Learning)
* [ML Geoscience](https://github.com/DIG-Kaust/MLgeoscience)
* [Be a Geoscience Detective](https://github.com/bluetyson/Be-a-geoscience-detective)
* [Earth ML](http://earthml.holoviz.org/tutorial/Machine_Learning.html) -> Some basic tutorials in PyData approaches
* [GeoMLA](https://github.com/thengl/GeoMLA) -> Machine Learning algorithms for spatial and spatiotemporal data
* [open-geospatial](https://github.com/opengeos/python-geospatial) -> Install multiple common packages at once


## Platforms

## Guides
* [Geospatial CLI](https://github.com/JakobMiksch/geospatial-cli) - List of geospatial command line tools 
* [Satellite Image Deep Learning](https://github.com/robmarkcole/satellite-image-deep-learning)
* [Earth Observation](https://github.com/RichardScottOZ/awesome-earthobservation-code)
* [Earth Artificial Intelligence](Awesome-Earth-Artificial-Intelligence)
* [Open Source GIS](https://link.springer.com/chapter/10.1007/978-3-030-53125-6_30) -> Comprehensive overview of the ecosystem

# Data Quality
* [Geoscience Data Quality for Machine Learning](https://github.com/RichardScottOZ/Geoscience-Data-Quality-for-Machine-Learning) -> Geoscience Data Quality for Machine Learning
* [Australian Gravity Data](https://github.com/RichardScottOZ/australia-gravity-data) -> Overview and analysis of gravity station data
* [Geodiff](https://github.com/MerginMaps/geodiff) -> Comparison of vector data
* [Redflag](https://github.com/agilescientific/redflag) -> Analysis of data and an overview to detect problems
# Machine Learning
* [CuML](https://github.com/rapidsai/cuml) -> CUDA accelerated
  * [scikit-learn acceleration notebook](https://colab.research.google.com/github/rapidsai-community/showcase/blob/main/getting_started_tutorials/cuml_sklearn_colab_demo.ipynb#scrollTo=fqJC2jtmqpva)
* [Dask-ml](https://github.com/dask/dask-ml) -> Distributed versions of some common ML algorithms
	* [paper](https://joss.theoj.org/papers/10.21105/joss.06889.pdf)
* [geospatial-rf](https://github.com/BritishGeologicalSurvey/geospatial-rf) -> Functions and wrappers to assist with random forest applications in a spatial context
* [Geospatial-ml](https://github.com/giswqs/geospatial-ml) -> Install multiple common packages at once

# Latent Space
* [Nested Fusion](https://github.com/pixlise/NestedFusion)
	* [paper](https://github.com/pixlise/NestedFusion/blob/main/24_KDD_Nested_Fusion_paper.pdf) -> Nested Fusion: Dimensionality Reduction and Latent Structure Analysis of Multi-Scale Nested Data for M2020 PIXL RGBU and XRF Data
## Metrics
* [scores](https://github.com/nci/scores) -> Verifying and evaluating models and predictions with xarray
## Probabilistic
* [NG Boost](https://github.com/stanfordmlgroup/ngboost) -> probabilistic regression
* [Probabilistic ML](https://github.com/ZhiqiangZhangCUGB/Probabilistic-machine-learning)
* [Bagging PU with BO](https://github.com/ZhiqiangZhangCUGB/Bagging-PU-with-BO) -> Positive Unlabeled Bagging with Bayesian Optimisation
## Clustering  
### Self Organising Maps
* [GisSOM](https://github.com/RichardScottOZ/GisSOM) -> Geospatial centred Self Organising Maps from Finland Geological Survey
  * [paper](https://www.lyellcollection.org/doi/pdf/10.1144/geochem2024-055) -> example of GisSOM example
* [SimpSOM](https://github.com/fcomitani/SimpSOM) -> Self Organising Maps 
### GPU Accelerated
* [GPU_PROCLUS](https://github.com/jakobrj/GPU_PROCLUS) -> gpu accelerated kmediods variant on subspaces
  * [GPU_PROCLUS](https://github.com/RichardScottOZ/GPU_PROCLUS) -> Windows 11 / MVCC version
### Self Organising Maps
* [TorchSOM](https://github.com/RichardScottOZ/TorchSOM) -> Self Organising Maps in Torch
* [aweSOM](https://github.com/tvh0021/aweSOM/tree/public-release) -> Accelerated Self-organizing Map (SOM) and Statistically Combined Ensemble (SCE)
  * [paper](https://joss.theoj.org/papers/10.21105/joss.07613)

### Other
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
	* [Cloud native data loaders for machine learning using Zarr and Xarray](https://earthmover.io/blog/cloud-native-dataloader)
* [zen3geo](https://github.com/weiji14/zen3geo) -> Xbatcher style data science with pytorch
### Explainability
* [Shap Values](https://github.com/slundberg/shap)
* [Weight Watcher](https://github.com/CalculatedContent/WeightWatcher) -> Analyse how well networks are trained
 * [weightwatcher.ai](https://weightwatcher.ai/)
 * [weightwatcher-ai.com](https://weightwatcher-ai.com/) -> Professional web version
### Self-supervised learning
* [Self Supervised](https://github.com/untitled-ai/self_supervised) -> Pytorch lightning implementations of multiple algorithms
* [Simclr](https://github.com/google-research/simclr)
* [Awesome self-supervised learning](https://github.com/jason718/awesome-self-supervised-learning) -> Curated list
### Hyperparameters
* [Hyperopt](https://github.com/hyperopt/hyperopt)
* [TPOT Automated ML](https://github.com/trhallam/tpot)
  
## Coding Environments
* [DEA Sandbox](https://docs.dea.ga.gov.au/setup/Sandbox/sandbox.html)
* [Cube In A Box](https://github.com/RichardScottOZ/cube-in-a-box)

# Community
* [Software Underground](https://softwareunderground.org/) - Community of people interested in exploring the intersection of the subsurface and code
  * [Chat Signup](https://softwareunderground.org/mattermost) - SWUNG community chat signup
  * [Mattermost](https://mattermost.softwareunderground.org/)- Community chat service
	* [Old Slack Channel](https://softwareunderground.org/slack)(deprecated, see mattermost above)
  * [Geoscience Open Source Tie-In](https://github.com/RichardScottOZ/gostin)
  * [Videos](https://www.youtube.com/c/SoftwareUnderground/videos)
    * [Transform 2022](https://www.youtube.com/playlist?list=PLgLft9vxdduDFkG9gtuNicNmb2YUzWqSQ)
  * [Awesome Open Geoscience](https://github.com/softwareunderground/awesome-open-geoscience )[note Oil and Gas Biased]
  * [Transform 2021 Hacking Examples](https://github.com/RichardScottOZ/Transform-2021)
  * [Segysak 2021 Tutorial](https://github.com/trhallam/segysak-t21-tutorial)
  * [T21 Seismic Notebook](https://github.com/stevejpurves/t21-seismic-notebook)
  * [Practical Seismic with Python](https://github.com/gmac161/practical-seismic-t21-tutorial)
  * [Transform 2021 Simpeg](https://github.com/simpeg/transform-2021-simpeg)
* [Pangeo](https://pangeo.io/)
  * [Forum](https://discourse.pangeo.io/)
  * [COG Best Practices](https://github.com/pangeo-data/cog-best-practices)
* [Digital Earth Australia](https://www.dea.ga.gov.au/)
  * [Slack Channel](https://opendatacube.slack.com/)
* [Open Source Geospatial Foundation](https://github.com/OSGeo/osgeo)
 * [OSGeoLive](https://github.com/OSGeo/OSGeoLive) -> Bootable DVD/USB with lots of open source geospatial software
* [ASEG](https://www.youtube.com/c/ASEGVideos/videos) -> videos from Australia Society of Exploration Geoscientists
* [AI for Geological Modelling and Mapping](https://www.youtube.com/@AI-GMM/videos) -> videos from the conference day
 * [conference](https://www.exeter.ac.uk/research/institutes/idsai/events/artificialintelligenceforgeologicalmodellingandmapping22-23may2024/)

# Cloud Providers
## AWS
* [ec2 Spot Labs](https://github.com/awslabs/ec2-spot-labs) -> Making automatically working sith Spot instances easier
* [Sagemaker Geospatial ML](https://aws.amazon.com/sagemaker/geospatial/)
* [Sagemaker](https://github.com/aws/amazon-sagemaker-examples) -> ML Managed Service
  * [SDK](https://github.com/aws/sagemaker-python-sdk)
  * [Entrypoint Utilities](https://github.com/aws-samples/amazon-sagemaker-entrypoint-utilities)
  * [Workshop 101](https://github.com/RichardScottOZ/sagemaker-workshop-101)
  * [Training Toolkit](https://github.com/aws/sagemaker-training-toolkit)
### Batch
* [Shepard](https://github.com/RichardScottOZ/shepard) -> Automated cloud formation setup of AWS Batch Pipelines: this is great
### Packages
* [Mlmax](https://github.com/awslabs/mlmax) - Start fast library
* [Smallmatter](https://github.com/aws-samples/smallmatter-package)
* [Pyutil](https://github.com/verdimrc/pyutil)
## General
* [Deep Learning Containers](https://github.com/aws/deep-learning-containers)
* [Loguru](https://github.com/Delgan/loguru) -> Logging library
* [AWS GDAL Robot](https://github.com/mblackgeo/aws-gdal-robot) -> Lambda and batch processing of geotiffs
* [Serverless Seismic Processing](https://github.com/vavourak/serverless_seismic_processing)
* [LIthops](https://github.com/lithops-cloud/lithops) -> multi-cloud distributed computing framework

# Overviews
* [Mineral Exploration](https://www.ga.gov.au/scientific-topics/minerals/mineral-exploration)
## Domains
* [Geology](https://en.wikipedia.org/wiki/Geology)
  * [Geologic Ages](https://en.wikipedia.org/wiki/Geologic_time_scale)
  * [Lithology](https://en.wikipedia.org/wiki/Lithology)
  * [Stratigraphy](https://en.wikipedia.org/wiki/Stratigraphy) 
* [Geochemistry](https://en.wikipedia.org/wiki/Geochemistry)
* [Geophysics](https://en.wikipedia.org/wiki/Geophysics)
* [Remote Sensing](https://en.wikipedia.org/wiki/Remote_sensing)

# Web Services
If listed it is assumed they are generally data, if just pictures like WMS it will say so.

## World
* [Critical Minerals and Deposits](https://portal.ga.gov.au/metadata/geochemistry/inorganic-geochemistry/critical-minerals-deposits-and-geochemistry/345cfeb7-9832-4c95-9f2e-59ec20cb1d91)
## Australia
* [AusGIN](https://www.geoscience.gov.au/web-services)
* [Geoscience Australia](http://services.ga.gov.au/)
 * [Mineral Potential](https://ecat.ga.gov.au/geonetwork/srv/api/records/f4bd18f3-5688-4c4d-8072-bdedc49a29e6) -> WMS
* [Geoscience Australia Catalogue Service](https://ecat.ga.gov.au/geonetwork/srv/eng/csw?request=GetCapabilities&service=CSW&acceptVersions=2.0.2&acceptFormats=application%2Fxml)
### Geology
* [AUSLAMP](http://services.ga.gov.au/gis/rest/services/AusLAMP_TISA_Stations/MapServer) - > Tennant Creek - MtIsa
* [Field Geology](https://services.ga.gov.au/gis/field-geology/wfs)
* [Deep Lithosphere](http://deep-lithospheric-structure.gs.cloud.ga.gov.au/ows) -> Deep Lithospheric Mineral Potential
* [Geochronology](http://geochronology-isotopes.gs.cloud.ga.gov.au/ows) -> Geochronology
* [Geological Provinces](https://services.ga.gov.au/gis/services/australian_geological_provinces/mapserver/wfsserver?version=2.0.0)
* [WMS](https://services.ga.gov.au/gis/rest/services/Australian_Geological_Provinces/MapServer) -> WMS picture
* [EGGS](http://services.ga.gov.au/gis/eggs/wms) -> Estimates of Geological and Geophysical Surfaces
* [Proterozoic Alkaline Rocks](https://services.ga.gov.au/gis/services/ProterozoicAlkalineAndRelatedIgneousRocksOfAustralia/MapServer/WFSServer) - Proterozoic Alkaline Rocks Dataset WFS {also has WMS}
 * [Cenozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/92b76dbc-95a7-4bf9-84f7-34d7602e66eb)
 * [Mesozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/e63b4771-cd18-4241-94b0-4e87708b4694)
 * [Paleozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/1f7163a7-12b0-44b1-bb8c-468641f34226)
 * [Archaean](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/ae37b73a-d5d6-4a48-a4db-4bb6bc0aec39)
* [Stratigraphy](https://services.ga.gov.au/gis/stratunits/ows) -> Stratigraphic Units
### Geophysics
* [Geophysics Surveys](https://services.ga.gov.au/gis/geophysical-surveys/ows?VERSION=2.0.0)
* [Seismic Surveys](https://services.ga.gov.au/gis/rest/services/Australian_Geological_Provinces/MapServer) -> Onshore seismic surveys
* [Magnetotelluric](https://services.ga.gov.au/gis/magnetotellurics/wfs) -> Northern Australia AUSLAMP Stations
### Other
* [Ni-Cu-PEGE](http://services.ga.gov.au/gis/services/MineralPotentialMapper/MapServer/WMSServer) -> Intrusion hosted Nickel Copper PGE Deposits
* [EFTF Area](http://services.ga.gov.au/gis/rest/services/ExploringForTheFutureProjectAreas/MapServer) -> Exploring for the future areas
* [Temperature](http://services.ga.gov.au/gis/rest/services/OZTemp_Interpreted_Temperature_5km_Depth/MapServer) -> Interpreted temperature
* [DEA](https://ows.dea.ga.gov.au/) -> Digital Earth Australia
 * [Land Cover](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146197)
 * [Waterbodies](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146197)
* [BOM](https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/groundwater/Bore_Hydrochemistry/FeatureServer) -> Bureau of Meteorology Hydrogeochemistry
### New South Wales
* [NSW](https://www.regional.nsw.gov.au/meg/geoscience/products-and-data/gis-web-services)
 * [WCS](https://gs.geoscience.nsw.gov.au/geoserver/ows?version=1.1.0)
 * [WFS Mineral Drillholes](https://gs.geoscience.nsw.gov.au/geoserver/ows?service=wfs&version=1.1.0&request=GetCapabilities)
 * [WFS Petroleum Drillholes](https://gs.geoscience.nsw.gov.au/geoserver/ows?service=wfs&version=1.1.0&request=GetCapabilities)
 * [WFS Coal Drillholes](https://gs.geoscience.nsw.gov.au/geoserver/ows?service=wfs&version=1.1.0&request=GetCapabilities)
 * [Seismic](https://gs-mv.geoscience.nsw.gov.au/geoserver/gsnsw/ows?version=1.0.0&typeName=gsnsw:dw_seismic_lines&outputFormat=shape-zip) -> Seismic and others
### Queensland
* [Queensland](https://gisservices.information.qld.gov.au/arcgis/rest/services)
 * [Geoscientific](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation) -> Geophysics and Report Index
 * [Geology](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation/GeologyDetailed/MapServer)
  * [Regional](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation/GeologyRegional/MapServer)
  * [State](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/GeoscientificInformation/GeologyState/MapServer)
 * [Tenements](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Economy/MineralTenement/FeatureServer)
 * [Roads](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/Transportation/Roads/MapServer)
 * [Watercourse](https://spatial-gis.information.qld.gov.au/arcgis/rest/services/InlandWaters/WaterCoursesAndBodies/MapServer)
 
### South Australia
* [SARIG](https://map.sarig.sa.gov.au/MapViewer/StartUp/?siteParams=WebServicesWidget)
 * [Drillholes](https://services.sarig.sa.gov.au/vector/drillholes/wfs?version=1.1.0)
 * [Geology](https://services.sarig.sa.gov.au/vector/geology/wfs?version=1.1.0)
 * [Geophysics](https://services.sarig.sa.gov.au/vector/geophysical_data/wfs?version=1.1.0)
 * [Prospectivity](https://services.sarig.sa.gov.au/raster/ProspectivityModelling/wms?version=1.1.1)
 * [Minerals and Mines](https://services.sarig.sa.gov.au/vector/mines_and_mineral_deposits/wfs?version=1.1.0)
 * [Remote Sensing](https://services.sarig.sa.gov.au/raster/RemoteSensing/wms?version=1.1.1)
 * [Seismic](https://services.sarig.sa.gov.au/vector/south_australia_seismic_data/wfs?version=1.1.0)
 * [Tenements](https://services.sarig.sa.gov.au/vector/mineral_tenements/wfs?version=1.1.0)
### Northern Territory
* [NTGS](http://geology.data.nt.gov.au/geoserver/wfs) -> Northern Territory Geological Survey
### Tasmania
* [Tasmania WFS](https://www.mrt.tas.gov.au/products/digital_data/web_feature_service)
* [Tasmania REST](https://data.stategrowth.tas.gov.au/ags/rest/services)
 * [Boreholes](https://data.stategrowth.tas.gov.au/ags/rest/services/MRT/Boreholes_and_Traces/MapServer)
### Victoria
* [Victoria Geonetwork](http://geology.data.vic.gov.au/)
### Western Australia
* [Western Australia](https://services.slip.wa.gov.au/public/services/SLIP_Public_Services/Industry_and_Mining_WFS/MapServer/WFSServer)
 * [Rest](https://services.slip.wa.gov.au/public/rest/services/SLIP_Public_Services)
## New Zealand
* [GNS](https://maps.gns.cri.nz/) -> List of web services
 
## South America
### Brazil
* [Brazil Geoportal](https://geoportal.cprm.gov.br/server/rest/services)
* [Brazil CPRM](https://geoportal.cprm.gov.br/image/rest/services)
### Peru
* [Ingement](https://geocatmin.ingemmet.gob.pe/arcgis/rest/services)
 * [Mineral Occurrences](https://geocatmin.ingemmet.gob.pe/arcgis/rest/services/SERV_OCURRENCIA_MINERAL/MapServer)
* [Environmental](https://geo.serfor.gob.pe/geoservicios/rest/services)
### Mexico
* [GeoInfo](https://mapasims.sgm.gob.mx/arcgis/rest/services) -> Rest services
### Argentina
* [SIGAM](https://sigam.segemar.gov.ar/https://sigam.segemar.gov.ar/geoserver217/wfs)
### Colombia
* [Rest](https://srvags.sgc.gov.co/arcgis/rest/services)
 * [Deposits 2018](https://srvags.sgc.gov.co/arcgis/rest/services/Mapa_metalogenico_2018/Mapa_Metalogenico_2018/MapServer)
### Uruguay
* [Rest](http://mapas.dinamige.gub.uy/arcgis/rest/services)
### Other
* [SIG Andes](http://mapsref.brgm.fr/wxs/1GG/SIGAndes_BRGM) -> Andes geology

## Europe
[EGDI](https://data.geus.dk/egdi/wfs/?whoami=modern_major_mineral@gmail.com&typenames=egdi_mineraloccurr_base_metals) -> EGDI Minerals
### Britain
* [BGS](https://ogcapi.bgs.ac.uk/collections?f=html) -> British Geological Survey
 * [Geoindex](https://mapapps2.bgs.ac.uk/geoindex/home.html?topic=Minerals&_ga=2.203824336.1992427000.1668901218-659606051.1668901218) -> mineral occurrence example
 * [Rest](https://map.bgs.ac.uk/arcgis/rest/services) -> BGS Rest services
 * [Inspire 625](http://ogc.bgs.ac.uk/digmap625k_gsml_insp_gs/wfs?service=WFS&request=GetCapabilities&AcceptVersions=2.0.0)
### Czech Republic
* [Rest](https://ags.cuzk.cz/arcgis/rest/services/zm/MapServer?f=json)
### Denmark
* [deus](http://data.geus.dk/geusmap/ows/25832.jsp?whoami=[email]) -> Greenland WMS/WFS
* [Deus](https://data.geus.dk/geusmap/ows/help/?mapname=oil_and_gas&epsg=25832)  - Update?
### Iceland
[web map](https://www.map.is/os/@656678,471081,z0,0)
### Ireland
* [Rest](https://gsi.geodata.gov.ie/server/rest/services)
 * [Mineral Locations](https://gsi.geodata.gov.ie/server/rest/services/Minerals/IE_GSI_Mineral_Locations_IE26_ITM/MapServer/0)
### Finland
* [GTK](https://www.gtk.fi/en/services/data-sets-and-online-services-geo-fi/map-services/) -> Geological Survey of Finland
  * [Finland](https://gtkdata.gtk.fi/arcgis/rest/services)
    * [Bedrock Geology](http://gtkdata.gtk.fi/arcgis/services/Rajapinnat/GTK_Kalliopera_WFS/MapServer/WFSServer_)
	* [Geophysics](http://gtkdata.gtk.fi/ArcGIS/services/Rajapinnat/GTK_Geofysiikka_WMS/MapServer/WMSServer)
	* [Ground Surveys](http://gtkdata.gtk.fi/arcgis/services/Rajapinnat/GTK_Pohjatutkimukset_WMS/MapServer/WMSServer)
* [Arctic Minerals](http://13.95.69.121:80/geoserver/erl/ows) -> Arctic 1M Mineral Occurrences
### Germany
* [BRG](https://services.bgr.de/uebersicht/kurzlinks)
### Hungary
* [MBFZ OGC](https://map.mbfsz.gov.hu/)
* [MBFZ Rest](https://map.mbfsz.gov.hu/arcgis/rest/services)
### Poland
* [Rest example](https://cbdgmapa.pgi.gov.pl/arcgis/rest/services/midas/MapServer/1?f=json) -> Many more mapservers
### Portugal
* [Portugal Geology](https://inspire.lneg.pt/arcgis/rest/services/CartografiaGeologica/CGP1M/MapServer)
 * [Mineral Occurrences](https://sig.lneg.pt/server/services/OcorrenciasMinerais/MapServer/WMSServer?request=GetCapabilities&service=WMS) -> WMS
 * [Cities and Town](https://inspire.lneg.pt/arcgis/rest/services/CartografiaGeologica/CGP1M/MapServer)
### Romania
* [IGR](https://inspire.igr.ro/geoserver/Test/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities) -> WMS only
 * [IGR minres](https://inspire.igr.ro/geoserver/minerals4eu/wms?SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities) -> WMS only
### Slovakia
* [Rest](https://gis.geology.sk/arcgis/rest/services)
### Spain
* [Spain](https://mapas.igme.es/gis/rest/services)
 * [Geology](https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geologico_200/MapServer) -> 200K
  * [1M](http://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geologico_1M/MapServer) -> 1M
  * [50K](https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_MAGNA_50/MapServer) -> 50K
  * [IGME Geode](https://mapas.igme.es/gis/rest/services/Cartografia_Geologica/IGME_Geode_50/MapServer)
 * [Geophysics](https://mapas.igme.es/gis/rest/services/BasesDatos/IGME_SIGEOF/MapServer)
 * [Copper](https://mapas.igme.es/gis/rest/services/AtlasGeoquimico/IGME_MapaIsovalores2012_Cu/MapServer) - Copper
 * [GeoFPI](https://mapas.igme.es/gis/rest/services/GeoFPI) - > Geology and Minerals South Portuguese Zone
 * [Water](https://mapas.igme.es/gis/rest/services/InfoIGME/AmbEsp_03/MapServer)
### Sweden
* [SGU Magnetics WMS](https://resource.sgu.se/service/wms/130/flyggeofysik-magnet)
* [SGU Uranium](https://resource.sgu.se/service/wms/130/flyggeofysik-gammastralning-uran)
* [SGU Borehole](https://api.sgu.se/oppnadata/borrhal/ogc/features/v1)
 * [Geophysics metadata](https://resource.sgu.se/dokument/produkter/geofysiska-flygmatningar-metadata-wms-beskrivning.pdf)
### Ukraine
* [Geoinform](https://geoinf.kiev.ua/wp/index.html) -> [currently suspended]

## North America
* [USGS Boreholes](https://services.arcgis.com/v01gqwM5QqNysAAi/ArcGIS/rest/services/NIBI_Borehole_Prod/FeatureServer/0)
### Canada
* [Manitoba](https://rdmaps.gov.mb.ca/arcgis/rest/services/MapGallery/MG_GEOLOGY_CLIENT/MapServer/)
* [Newfoundland & Labrador](https://dnrmaps.gov.nl.ca/arcgis/rest/services/GeoAtlas/Map_Layers/MapServer)
* [NWT](https://services3.arcgis.com/GSr8HAQhtEt4sNnv/arcgis/rest/services/)
  * [Rest](https://www.maps.geomatics.gov.nt.ca/Geocortex/Essentials/REST/sites/Spatial_Data_Warehouse/map)
  * [References](https://hosting.wsapi.cloud.bom.gov.au/arcgis/rest/services/groundwater/Bore_Hydrochemistry/FeatureServer)
* [Quebec](https://servicesvectoriels.atlas.gouv.qc.ca/IDS_SGM_WMS/service.svc/get)
* [Yukon](https://mapservices.gov.yk.ca/arcgis/rest/services/GeoYukon/GY_Geological/MapServer)
### USA
* [USGS World Mineral](https://mrdata.usgs.gov/services/wfs/ofr20051294?version=1.1.0)
* [USGS MRDS](https://mrdata.usgs.gov/services/mrds?request=getcapabilities&service=WFS&version=1.0.0&)
* [Minnesota](https://mngs-umn.opendata.arcgis.com/)
### Asia
* [China](http://data.ngac.org.cn/mineralresource/index.html?id=302c137ee126465095b3df8e68168d8c) -> WMS mineral deposit wap
 * [orefield](http://219.142.81.85/arcgis/rest/services/矿产地数据库2019/orefield2019/MapServer/0) -> Mineral occurrence points
* [India Mineral](https://bhukosh.gsi.gov.in/arcgis/services/Mineral/Mineral/MapServer/WmsServer) -> WMS
* [India Geophysics](https://bhukosh.gsi.gov.in/wms/en)
* [Korea](https://data.kigam.re.kr/mgeo/geoserver/ows?service=wfs&version=1.0.0&request=GetCapabilities)
* [ASEAN wms](https://geohazards-info.gsj.jp/amdis/index.php) -> no data, just picture
### Africa
* [Africa Geoportal](https://services8.arcgis.com/oTalEaSXAuyNT7xf/ArcGIS/rest/services) -> Rest services
* [Africa 10M](http://mapsref.brgm.fr/wxs/1GG/SIGAfrique_BRGM_Africa_MineralResources) -> Africa 10M Mineral Occurrences
https://pubs.usgs.gov/of/2005/1294/e/OF05-1294-E.pdf
* [IPIS Artisanal Mines](http://geo.ipisresearch.be/geoserver/wfs) - > There is a WMS version too
 * [github](https://github.com/IPISResearch)
* [Uganda](https://gmis.beak.de/geoserver/uganda/wms?VERSION=1.3.0) -> GMIS WMS

### General
* [Mineral Exploration Web Services](https://github.com/jack-tuna/Mineral_Exploration_Web_Services) -> QGIS plugin with access to many relevant web services

## Other
* [Open Street Map](https://tile.openstreetmap.org/{z}/{x}/{y}.png) -> useful general tile service

# APIs
* [Open Data API](https://github.com/RichardScottOZ/open-data-api) -> GSQ Open Data Portal API
  * [Geochemistry parsing](https://github.com/geological-survey-of-queensland/geochemistry_parsing)
* [CORE](https://core.ac.uk/data) -> Open Research Texts
  * [API Notebook](https://colab.research.google.com/drive/1_bjqDQhqj7AnSfoCAXDCMOnGZLPQWKfu?usp=sharing) -> Example and fucntions
* [SHARE](https://share.osf.io/discover?q=mineral%20AND%20exploration) -> Open Science API
* [USGS Publications](https://pubs.er.usgs.gov/documentation/web_service_documentation)
* [CROSSREF](https://api.crossref.org/swagger-ui/index.html)
* [xDD](https://xdd.wisc.edu/api/v1) -> former GeoDeepDive
 * [ADEPT](https://xdd.wisc.edu/adept/) -> GUI to xDD to search 15M harvested papers
* [OpenAlex](https://docs.openalex.org/api)
  * [api](https://api.openalex.org/)
  * [diophila Python Library](https://github.com/smierz/diophila)
  * [Python Library](https://github.com/dpriskorn/OpenAlexAPI)
 	* [paper](https://arxiv.org/abs/2205.01833)
* [Macrostrat](https://macrostrat.org/api)
	* [paper](https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.189) 
* [OpenMinData](https://github.com/ChuBL/OpenMindat) -> facilitate querying and retrieving data on minerals and geomaterials from the Mindat API
* [Iceland Geological Society](https://api.orkustofnun.is/)
 
 # Data Portals
## World
* [Earth Model Collaboration](https://ds.iris.edu/ds/products/emc/) -> access to various Earth models, visualization tools for model preview, facilities to extract model data/metadata and access to the contributed processing software and scripts.
* [ISC Bulletin](https://www.isc.ac.uk/iscbulletin/search/fmechanisms/) -> Earthquake focal mechanism search
* [Magnetics Information Consortium](https://www2.earthref.org/MagIC/search) -> paleomagnetic, geomagnetic, rock magnetic
* [Earthchem](https://www.earthchem.org/) -> Community-driven preservation, discovery, access, and visualization of geochemical, geochronological, and petrological data
## Australia
### Geoscience Australia
* [Geoscience Australia Data Catalogue](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/)
 * [AusAEM](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/search?resultType=details&sortBy=relevance&fast=index&_content_type=json&from=1&to=20&title_OR_altTitle_OR_any=AusAEM)
* [Geoscience Australia Portal](https://portal.ga.gov.au/)
 * [Exploring for the Future Portal](https://portal.ga.gov.au//eftf) -> Geoscience Australia web portal with download information
  * [AusAEM](https://www.eftf.ga.gov.au/ausaem)
  * [AusLAMP](https://www.eftf.ga.gov.au/auslamp)
 * [Geochronology and Isotopes](https://portal.ga.gov.au/persona/geochronology)
 * [Hydrogeology Catchments](https://portal.ga.gov.au/) -> search for catchments layer
 * [Critical Minerals Mapping Initiative](https://portal.ga.gov.au/persona/cmmi)
 * [Australian Stratigraphic Units](https://asud.ga.gov.au/search-stratigraphic-units)
  * [Australian Borehole Stratigraphic Units](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147641) -> Compilation for groundwater of sedimentary units
 * [Geoscience Australia Geophysics thredds](https://dapds00.nci.org.au/thredds) -> OpendDAP and https access
 * [MORPH gdb](https://github.com/Neil-Symington/MORPH_gdb) -> Officer Musgrave drilling data
### CSIRO
* [CSIRO Data Access Portal](https://data.csiro.au/)
 * [Regolith Depth](https://data.csiro.au/collection/csiro:11393)
 * [TWI](https://data.csiro.au/collection/csiro:5588?_st=browse&_str=1&_si=1&browseType=kw&browseValue=Topographic%20Wetness%20Index) -> Topographic Wetness Index
* [ASTER Geoscience Maps](https://confluence.csiro.au/public/SpecSens/aster-geoscience-maps) -> Website
	* [FTP](ftp://ftp.arrc.csiro.au/arrc/Australian_ASTER_Geoscience_Map/) -> CSIRO ftp site
	* [ASTER Maps notes](https://confluence.csiro.au/public/SpecSens/files/276430859/276430921/1/1426138425933/Australian+ASTER+Geoscience+Product+Notes+FINALx.pdf) -> Notes for the above
### AuScope
* [3D Geology](http://geomodels.auscope.org.au/) -> Models from multiple areas
### TERN
* [Enhanced Bare Earth Covariates for Soil and Lithological Modelling](https://portal.tern.org.au/metadata/21910)
### Bureau of Meteorology
* [Groundwater Explorer](http://www.bom.gov.au/water/groundwater/explorer/map.shtml) -> Bureau of Meteorology
### Foundational Spatial Data
* [Elvis](https://elevation.fsdf.org.au/)
### South Australia
* [SARIG](https://map.sarig.sa.gov.au/) -> South Australia Geological Survey geospatial map based search
* [SARIG Catalogue](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/search) -> data catalogue
  * [3D Models](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/search?facet.q=type%2Fmodel&resultType=details&sortBy=popularity&from=1&to=20&fast=index&_content_type=json)
	* [SA Wide](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/cat0/Record?w=catno%3D2036881+OR+catno%3D2036883+OR+catno%3D2036887&sid=c2ca73f5685c43249e6c1ab5ec7dc2e5&set=1&m=3)
	 * [s3 location](https://dsd-gdp.s3.amazonaws.com/GDP00022.zip)
  * [Data Packages](https://dem-sdp.s3-ap-southeast-2.amazonaws.com/index.html) - Annual update
  * [s3 Reports](http://mer-env.s3-website-ap-southeast-2.amazonaws.com/) -> Reports and textracted versions in s3 bucket with web interface)
  * [Reports](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/samref/sarig1/cat0/MSearch;jsessionid=492C6538B64080CE8B13E91C79F8B1BA)
  * [Seismic](https://www.petroleum.sa.gov.au/data-centre/seismic-data)
    * [Seismic downloads](https://sarigbasis.pir.sa.gov.au/WebtopEw/ws/segy2d/web/segy/ResultSet?siblingtreeid=e6d6d3af10d149d39ba2141b9d1ce660&sid=6fac787ca578408bad6cfb514eb15498&order=NATIVE%28%27LINE%2Fascend%27%29&rpp=-1&set=1&reorder=1&bclabel=Result+Set) -> One page of links
### Northern Territory
* [STRIKE](https://strike.nt.gov.au/wss.html) -> Northern Territory Geological Survey
* [GEMIS](https://geoscience.nt.gov.au/gemis/)
  * [McArthur Basin](https://geoscience.nt.gov.au/gemis/ntgsjspui/handle/1/81751) -> 3D Model
  * [Geophysical Surveys](https://geoscience.nt.gov.au/downloads/NTWideDownloads.html)
  * [Geophysics](https://geoscience.nt.gov.au/gemis/ntgsjspui/handle/1/81428) -> reference
  * [Drilling and Geochemistry](https://geoscience.nt.gov.au/gemis/ntgsjspui/handle/1/81743) -> reference
	  * [Data Package](https://geoscience.nt.gov.au/gemis/ntgsjspui/bitstream/1/81743/2/DIP001.zip) -> data package
### Queensland
* [Geological Survey of Queensland](https://geoscience.data.qld.gov.au/)
* [Geophysical Surveys](http://qldspatial.information.qld.gov.au/catalogue/custom/search.page?q=%22Airborne%20geophysical%20survey%20-%20Queensland%22)
* [Drilling and geochemistry](https://geoscience.data.qld.gov.au/data/geochemistry/whole-of-queensland-geochemistry-databases)
### Western Australia
* [GEOVIEW](https://geoview.dmp.wa.gov.au/geoview/?Viewer=GeoView) -> Western Australia Geological Survey
* [DMIRS](https://dasc.dmirs.wa.gov.au/) -> DMIRS Data and Software Centre
 * [Download URLS](https://dasc.dmirs.wa.gov.au/Download/File/3599) -> dataset of download links
* [Drilling and Geochemistry](https://wamexgeochem.net.au/)
 * [Download package](https://exp-gswa-mdhdb-bkt01.s3.ap-southeast-2.amazonaws.com/GSWA_Nov2022_FlatTables.zip) - improvement?
 * [Geochemistry](https://wamexgeochem.net.au/)
 * [Petroleum Wells with depths](https://wapims.dmp.wa.gov.au/WAPIMS/)
* [data WA subset](https://catalogue.data.wa.gov.au/org/department-of-mines-industry-regulation-and-safety)
## NSW
* [MINVIEW](https://minview.geoscience.nsw.gov.au/) -> New South Wales Geological Survey
* [DiGS](https://search.geoscience.nsw.gov.au/) -> Publications and Geotechnical collections
## Tasmania
* [MRT](https://www.mrt.tas.gov.au/products)
* [MRT Maps](https://www.mrt.tas.gov.au/mrt_maps/app/list/map) -> Webmap
## Victoria
* [Earth Resources](https://earthresources.vic.gov.au/geology-exploration/maps-reports-data)
  * [Geophysics](http://earthresources.efirst.com.au/categories.asp?cID=13)
* [GeoVIC](https://earthresources.vic.gov.au/geology-exploration/maps-reports-data/geovic) -> Webmaps needs registration to be more useful
## New Zealand
* [Exploration Database](https://data.nzpam.govt.nz/GOLD/system/mainframe.asp) -> Online
* [GERM](https://data.gns.cri.nz/germ/submitQuery.html) -> Geological Resource Map of New Zealand
* [Geology](https://data.gns.cri.nz/geology/) -> Web Map
  * https://maps.gns.cri.nz/gns/wfs 

## South America
### Argentina
* [SIGAM](https://sigam.segemar.gov.ar/wordpress) -> Argentina Geological Survey
 * [SIGAM](https://sigam.segemar.gov.ar/wordpress/geoservicios/)
### Brazil
* [CPRM](https://www.cprm.gov.br/en/Geology-53) -> Brazil Geological Survey
 * [Downloads](https://geosgb.cprm.gov.br/geosgb/downloads_en.html) -> Brazil Geological Survey Downloads
* [Rigeo](https://rigeo.cprm.gov.br/) -> Institutional Repository of Geosciences
### Chile
* [Portalgeomin](https://portalgeomin.sernageomin.cl/)
### Colombia
* [Geportal](https://srvags.sgc.gov.co/JSViewer/Recursos_Minerales_Ingles/)
### Mexico
* [GeoInfo](https://www.sgm.gob.mx/GeoInfoMexGobMx/)
### Paraguay
* [Geology of Paraguay](https://www.geologiadelparaguay.com.py/Mapas.htm)
### Peru
* [Ingemmet GeoPROMINE](https://ingemmet-peru.maps.arcgis.com/apps/webappviewer/index.html?id=95abc980a2ad441191fde60d66266d2b) -> Geological Survey of Peru
 * [GeoMAPE](https://ingemmet-peru.maps.arcgis.com/apps/webappviewer/index.html?id=6581aa545eca4547acdc1fc7136f1fcd)
### Uruguay
* [Dimage](http://visualizadorgeominero.dinamige.gub.uy/DINAMIGE_mvc2/)

## Europe
* [EGDI](https://www.europe-geology.eu/metadata/) -> Europe geoscience
 * [WFS](https://data.geus.dk/egdi/wfs/help/?layers=egdi_mineraloccurr_base_metals)
 * [Promine](https://servicesvectoriels.atlas.gouv.qc.ca/IDS_SGM_WMS/service.svc/get)
* [Inspire](https://inspire-geoportal.ec.europa.eu/) -> Inspire Geoportal
### Denmark
* [Danish Subsurface Data](https://data.geus.dk/geusmap/?mapname=oil_and_gas&lang=en#baslay=&optlay=&extent=-1384000,5329054.6875,2426000,7120945.3125&layers=samba_wellbores,dkskaermkort&filter_0=txt_search.part%3D%26status.part%3D)
### Finland
* [Minerals4EU](http://minerals4eu.brgm-rec.fr)
* [GTK](https://www.gtk.fi/en/services/data-sets-and-online-services-geo-fi/) -> Geological Survey of Finland
 * [Geochemical Maps](http://weppi.gtk.fi/publ/foregsatlas/maps_table.php) -> pdf only!
### Sweden
* [SGU](https://www.sgu.se/en/products/geological-data/use-data-from-sgu/) -> Swedish Geological Survey
### Spain
* [IGME](http://info.igme.es/catalogo/catalog.aspx?catalog=2&shfo=false&shdt=false&master=infoigme&lang=spa&types=4) -> Spanish Geological Survey
### Portugal
* [Geoportal](https://geoportal.lneg.pt/)
 * [Mineral Occurences](https://geoportal.lneg.pt/umbraco/SiorminpPlugin/SiorminpApi/GetDetails?id=2200)
### Ireland
* [GSI](https://www.gsi.ie/en-ie/data-and-maps/Pages/default.aspx) -> Geological Survey of Ireland
 * [GSI](https://dcenr.maps.arcgis.com/apps/MapSeries/index.html?appid=a30af518e87a4c0ab2fbde2aaac3c228) - Map viewer
* [Goldmine](https://secure.decc.gov.ie/goldmine/index.html) -> Map and document search
* [data.gov.ie](https://data.gov.ie/organization/geological-survey-of-ireland) -> National portal view
* [isde](https://isde.ie/geonetwork/srv/eng/catalog.search#/) -> Irish Spatial Data Exchange
### Norway
* [NGU](https://www.ngu.no/prospecting/) -> Norway Geological Survey
 * [database](https://aps.ngu.no) -> Mineral resources and stratigraphy lookups
 * [github](https://github.com/ngu)
 * [API](https://www.kartverket.no/en/api-and-data)
 * [Geoporta](https://geo.ngu.no/geoscienceportalopen/search) -> Geophysics
* [GEONORGE](https://geoportal.lneg.pt/umbraco/SiorminpPlugin/SiorminpApi/GetDetails?id=40) -> Data catalogue with download
### Britain
* [Britain](https://www2.bgs.ac.uk/mineralsuk/maps/maps.html)
 * [mapserver](https://mapapps2.bgs.ac.uk/geoindex/home.html?topic=Minerals)
 * [github](https://github.com/orgs/BritishGeologicalSurvey/)
### Ukraine
* [Mineral Resources](https://eng.minerals-ua.info/)
### Russia
* [Russian Geological Research Institute](https://www.vsegei.ru/en/) -> Inaccessible currently
* [RGU](https://rfgf.ru/map/) -> GIS project of deposits
### Germany
* [Geoportal](https://geoportal.bgr.de/mapapps/resources/apps/geoportal/index.html?lang=en#/)
 * [Geomap](https://geoportal.bgr.de/mapapps/resources/apps/geoportal/index.html?lang=en#/geoviewer) -> M
 * [Atom](https://services.bgr.de/atomfeeds/service.xml) -> Atom data feed
 * [GDI](https://gst.bgr.de) -> 3D Models Germany
### France
* [Infoterre](https://infoterre.brgm.fr/viewer/MainTileForward.do) -> French Geological Survey
### Croatia
* [Geoportal](https://www.hgi-cgs.hr/geoloske-karte/#) -> Croatia Geological Survey
* [Geology](http://webgis.hgi-cgs.hr/gk300/default.aspx)
### Czech Republic
* [GS](http://www.geology.cz/extranet/mapy/mapy-online/mapove-aplikace) -> Czech Geological Survey
### Slovenia
* [Data Catalogue](https://egeologija.si/geonetwork/srv/eng/catalog.search#/)
### Slovakia
* [Data Catalogue](https://apl.geology.sk/mapportal/#/aplikacie/3/)
* [Geoportapi api](https://apl.geology.sk/geoportal/#searchPanel)
### Hungary
* [MBFSZ Maps](https://map.mbfsz.gov.hu/)
### Romania
* [IGR](https://geoportal.igr.ro/viewminres) -> Romania Geological Survey
 * [Mineral Resources](https://geoportal.igr.ro/viewminres)
### Poland
 * [Geoportal](https://geoportal.pgi.gov.pl/portal/page/portal/PIGMainExtranet)
### United Kingdom
* [UK Onshore Geophysical Library](https://ukogl.org.uk/)
* [OS Data Hub British Geology](https://osdatahub.os.uk/downloads/open)
 * [Geology 625](https://osdatahub.os.uk/downloads/open/BGS_Geology_625k)

 
## North America
### Canada
* [Natural Resources Canada](https://www.nrcan.gc.ca/earth-sciences/geography/atlas-canada/explore-our-data/16892)
 * [github](https://github.com/NRCan)
 * [Geoscience Data Repository](https://gdr.agg.nrcan.gc.ca/) -> DAP Server
 * [Mining Web Map Portal](https://osdp-psdo.canada.ca/dp/en/explore?search_event=dev_mining)
 * [DEM](https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333) -> Canada DEM in COG format
 * [CDEM](https://open.canada.ca/data/en/dataset/7f245e4d-76c2-4caa-951a-45d1d2051333) -> Digital Elevation Model (2011)
* [Ontario](https://www.geologyontario.mndm.gov.on.ca/ogsearth.html)
  * [Geology Ontario](https://www.hub.geologyontario.mines.gov.on.ca/)
  * [Drilling](https://www.geologyontario.mndm.gov.on.ca/mines/ogs/databases/OMI.zip)
* [Quebec](https://gq.mines.gouv.qc.ca/documents/SIGEOM/TOUTQC/ANG/)
 * [SIGEOM Database](https://sigeom.mines.gouv.qc.ca/signet/classes/I1102_aLaCarte?l=F)
  * [Drilling](https://gq.mines.gouv.qc.ca/documents/SIGEOM/TOUTQC/ANG/GPKG/SIGEOM_QC_Drilling_GPKG.zip)
* [British Columbia](https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/british-columbia-geological-survey/publications/digital-geoscience-data)
 * [Mineral occcurrence database](https://catalogue.data.gov.bc.ca/dataset/minfile-mineral-occurrence-database)
* [Yukon](https://data.geology.gov.yk.ca/)
  * [ftp](https://ygsftp.gov.yk.ca/)
* [Nova Scotia](https://novascotia.ca/natr/meb/maps/)
 * [provincial](https://novascotia.ca/natr/meb/download/gis-data-maps-provincial.asp)
* [Prince Edward Island](http://www.gov.pe.ca/gis/download.php3)
* [Saskatchewan](https://geohub.saskatchewan.ca) 
 * [Mineral occurrence database](https://applications.saskatchewan.ca/Apps/ECON_Apps/dbsearch/MinDepositQuery/default.aspx?ID=5987)
* [Newfoundland](https://geoatlas.gov.nl.ca/Default.htm) -> didn't work in Chrome, tried it in Edge
* [New Brunswick](https://www2.gnb.ca/content/gnb/en/departments/erd/open-data/metallic-minerals.html#3)
  * [Exploration drillholes](https://hub.arcgis.com/datasets/150eb308c438443e9d8666c3c730c969_0/about?locale=en)
* [Alberta](https://ags.aer.ca/)
 * [Interactive Mapping Application](https://experience.arcgis.com/experience/d813bc54fcde4e099de7b399f7145434/)
* [Northwest Territories](https://app.nwtgeoscience.ca/)
 * [Mineral Tenure](https://www.maps.geomatics.gov.nt.ca/)
* [Nunavut Geoscience](https://nunavutgeoscience.ca/apps/showing/showQuery.php)
### USA
* [USGS](https://www.sciencebase.gov/catalog/item/5888bf4fe4b05ccb964bab9d) -> Map database
 * [MRDS](https://mrdata.usgs.gov/mrds/) -> Mineral Resources Data Systems
* [Earth Explorer](https://earthexplorer.usgs.gov) -> USGS Remote Sensing Data Portal
* [National Map Database](https://ngmdb.usgs.gov/ngmdb/ngmdb_home.html)
 * [National Map Database](http://ngmdb.usgs.gov/maps/mapview/)
 * [Alaska](https://pubs.er.usgs.gov/publication/sim3340)
* [ReSci](https://www.sciencebase.gov/catalog/item/4f4e4760e4b07f02db47dfb4) -> Registry of Scientific Collections of the National Geological and Geophysical Data Preservation Program
* [Michigan](https://geo.btaa.org/)
## Africa
* [Cadastre](https://landadmin.trimble.com/cadastre-portals/)
* [Hydrogeology](https://ggis.un-igrac.org/maps/1900) -> Hydrogeology and geology from groundwater atlas
* [West Africa](https://ars.els-cdn.com/content/image/1-s2.0-S0301926815001771-mmc1.csv) -> Mineral deposits
* [Namibia](https://namibia.africageoportal.com/)
  * (https://portal.mme.gov.na/page/MapPublic)
 * [Mineral Occurrences](https://namibia.africageoportal.com/datasets/2eee87a9bec944b1b628aad36c262407_0/explore?location=55.319481%2C5.326328%2C7.67&showTable=true)
 * [Miners](https://services8.arcgis.com/oTalEaSXAuyNT7xf/ArcGIS/rest/services/Namibia_Mines/FeatureServer/layers)
* [South Africa](https://maps.geoscience.org.za/portal/apps/sites/#/council-for-geoscience-interactive-web-map-1) -> South Africa geological survey
 * [Mineral Occurrences](https://maps.geoscience.org.za/download/mineral-shapefiles.php) -> Example where you need to log in to download
* [Uganda](https://gmis.beak.de/uganda/) -> GMIS portal
 * [Metallic minerals](http://catalog.data.ug/dataset/metallic-minerals)
* [Tanzania](https://www.gmis-tanzania.com/)
 * [Mineral Occurrences](https://www.gmis-tanzania.com/download/minocc.zip)
 * [Mines](https://www.gmis-tanzania.com/download/mines.zip)
* [SIGM](http://41.224.38.194:8080/SIGM/pages/geocatalogue/geocatalogue.xhtml) -> Tunisia Geology and Mining
* [Zambia](https://portals.landfolio.com/zambia/) -> Zambia tenements here
  * [Kobold geophysics grids](https://www.kaggle.com/code/tylerhowe/regridded-kobold-zambia-data)
## Asia
### ASEAN
* [Mineral Occurrences](https://geohazards-info.gsj.jp/main/)
### China
* [Geoscientific Data](http://dcc.ngac.org.cn/en?ssoguid=d7b8fe124bdf4f02b0f0b63bde4db79b)
* [Mineral Occurrences](http://dcc.cgs.gov.cn/cn/geologicalData/details/doi/10.23650/data.A.2019.NGA120157.K1.1.1.V1)
 * [National Mineral Deposit Database](http://dcc.ngac.org.cn/en//geologicalData/details/doi/10.23650/data.C.2018.NGA120770.K1.1.1.V1)
### Indonesia
* [Sedimentary Basins](https://geology.esdm.go.id/geomigas/)
### India
* [Bhukosh](https://bhukosh.gsi.gov.in/Bhukosh/Public) -> India Geological Survey
 * Note Rajasthan geology doesn't work except piecemeal which is painful - if you want it, let me 
### Japan
* [IGG Data] (https://unit.aist.go.jp/igg/en/database/index.html)
### Korea
* [geoporta](http://data.kigam.re.kr/search)
### Saudi Arabia
* [National Geological Database Portal](https://ngdp.sgs.gov.sa/ngp/) 
### Thailand
* [GIS Portal](https://gis.dmr.go.th/DMR-GIS/gis)
## Other
### Geology
* [StratDB](https://sil.usask.ca/stratdb-data-compilation.php)
* [GEM Global Active Faults](https://github.com/GEMScienceTools/gem-global-active-faults)
* [RRuff Mineral Properties](https://rruff.info/ima/)
 * [article](https://hazen.carnegiescience.edu/research/evolutionary-system-mineralogy) -> Evolutionary system of mineralogy
* [OneGeology](http://onegeology.brgm.fr/OnegeologyGlobal/)
 * [catalog](http://onegeology-geonetwork.brgm.fr/geonetwork3/srv/eng/catalog.search#/search)
## Iran
### Geology
* [OGC BGS 1M Bedrock](https://ogc.bgs.ac.uk/cgi-bin/BGS_GSI_EN_Bedrock_and_Structural_Geology/ows)
### General
* [OSF](https://osf.io/) -> Open Science Foundation
 * [Sediment Hosted Base Metals](https://osf.io/twksd/) -> Sediment Hosted Base Metals
 * [Lithosphere Athenosphere Boundary](https://osf.io/twksd/) -> LAB Hoggard/Czarnota
* [Geological Survey list](https://mineralplatform.eu/investment/exploration-mining-opportunities/geological-survey)


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
* [Sustainable Minerals Institute](https://smi.uq.edu.au/programs) -> Queensland organisation of university affiliated researchers producing datasets and knowledge
### Canada
* [British Columbia](https://www2.gov.bc.ca/gov/content/industry/mineral-exploration-mining/british-columbia-geological-survey/publications/digital-geoscience-data#ARIS)
  * [Search](https://aris.empr.gov.bc.ca/search.asp?mode=request&newsearch=Y) -> Mineral Assessment Reports
  * [Publications](http://webmap.em.gov.bc.ca/mapplace/minpot/Publications_Report.asp) -> Publications
* [Ontario](https://data.ontario.ca/en/dataset/assessment-files) -> Mineral Asssessment Reports
  * [Ontario Publications](https://www.geologyontario.mndm.gov.on.ca/Publications_Description.html)
* [Alberta](https://content.energy.alberta.ca/minerals/abmarsv2/?err=se)
  * [Publications](https://ags.aer.ca/products/all-publications?title=&report-id=&publication_type=All&sort_by=created&sort_order=DESC&page=0)
* [Yukon](https://data.geology.gov.yk.ca/AssessmentReports)
 * [Footprint](https://www.arcgis.com/home/item.html?id=49cc2473b4904ced9dbd530944f5d2e1)
* [Manitoba](https://www.manitoba.ca/iem/mines/assess.html)
  * [Geosciencetific Maps](https://rdmaps.gov.mb.ca/Html5Viewer/index.html?viewer=MapGallery_Geology.MapGallery)
  * [Publications](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146278)
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
 * [json](https://data.michigan.gov/api/views/8zkk-z5n4/rows.json?accessType=DOWNLOAD)
* [Alaska](https://dggs.alaska.gov/pubs)
* [Washington](https://www.dnr.wa.gov/publications/ger_publications_list.pdf)
### Other
* [British Geological Survey NERC](https://nora.nerc.ac.uk)
  * [Mineral Potential](https://www2.bgs.ac.uk/mineralsuk/exploration/potential/mrp.html)
  * [Search](https://nora.nerc.ac.uk/cgi/facet/archive/simple2?screen=XapianSearch&dataset=archive&order=&q=Mineral+AND+exploration&_action_search=Search )
  * [API example](https://nora.nerc.ac.uk/cgi/facet/archive/simple2/export_nerc_JSON.js?screen=XapianSearch&dataset=archive&_action_export=1&output=JSON&exp=0%7C1%7C%7Carchive%7C-%7Cq%3A%3AALL%3AIN%3AMineral+AND+exploration%7C-%7C&n=&cache=)
  * [Publications](https://www.bgs.ac.uk/information-hub/publications/)
  * [MEIGA](https://www.bgs.ac.uk/news/over-600-mineral-exploration-project-reports-now-available-through-the-uk-critical-minerals-intelligence-centre/) -> MEIGA 600 BGS mineral exploration project reports
* [GeoLagret](https://www.sgu.se/en/products/search-tools/geolagret/exploration-reports/) -> Sweden
* [MinData](https://www.mindat.org/mineralindex.php) -> Compilation of rock locations from around the world
* [Mineral Database](https://rruff.info/ima/) -> Exportable list of minerals with scientific properties and ages
* [NASA](https://www.sti.nasa.gov/research-access/)  
* [ResearchGate](https://www.researchgate.net/) -> Researcher and professional network

# Tools
## GIS
  * [QGIS](https://qgis.org/en/site/) -> GIS Data Visualisation and Analysis Open Source desktop application, has some ML tools : Indispensible for some quick and easy viewing
    * [2D Geology in QGIS](https://github.com/frizatch/2DGeology_in_QGIS) -> Workshop for QGIS NA 2020 introducing geologic maps and cross-sections for students and hobbyists
    * [OpenLog](https://gitlab.com/geolandia/openlog/openlog-qgis-plugin) -> Drillhole plugin beta
    * [Geo-SAM](https://github.com/coolzhao/Geo-SAM) -> QGIS plugin for Segment Anything with rasters
    * [QGIS Project Packager](https://github.com/gbrlpzz/qgis_project_packager)
	* [Weights-of-Evidence](https://github.com/chudasama-bijal/QGIS-Plugin-Weights-of-Evidence-Model)
	 * [plugin](https://plugins.qgis.org/plugins/wofe_module/)
  * [GRASS](https://github.com/OSGeo/grass) 
  * [saga](https://github.com/saga-gis/saga-gis) -> mirror of sourceforge
## 3D
* [PyVista](https://github.com/pyvista/pyvista) -> VTK wrapping api for great data visualisation and analysis
  * [PVGeo](https://pvgeo.org/index.html)
  * [Pyvista-Xarray](https://github.com/RichardScottOZ/pyvista-xarray) -> Transforming xarray data to VTK 3D painlessly: a great library!
  * [OMFVista](https://github.com/OpenGeoVis/omfvista) -> Pyvista for Open Mining Format
  * [Scipy 2022 Tutorial](https://github.com/pyvista/pyvista-tutorial)
* [TorchMesh](https://github.com/peterdsharpe/torchmesh) -> GPU accelerated Mesh programming with PyVista integration
* [PyMeshLab](https://github.com/cnr-isti-vclab/PyMeshLab) -> Mesh transformation
* [Open Mining Format](https://github.com/gmggroup/omf)
* [Whitebox Tools](https://github.com/jblindsay/whitebox-tools)
  * [GUI](https://github.com/giswqs/whiteboxgui) -> Desktop version
* [Subsurface](https://github.com/softwareunderground/subsurface)
* [Geolambda](https://github.com/bluetyson/geolambda) -> AWS Lambda setup
* [Geoscience Analyst](https://mirageoscience.com/mining-industry-software/geoscience-analyst/)
  * [geoh5py](https://geoh5py.readthedocs.io/) -> getting data to and from geoh5 projects
  * [geoapps](https://geoapps.readthedocs.io/en/stable/) -> notebook based applications for geophysics via geoh5py
  * [geoh5vista](https://github.com/derek-kinakin/geoh5vista)
  * [gams](https://github.com/eroots/gams) -> magnetic data analysis
 	* [paper](https://www.earthdoc.org/content/journals/10.3997/1365-2397.fb2024019) - A Framework for Mineral Geoscience Data and Model Portability - geoh5

* [Rayshader](https://github.com/tylermorganwall/rayshader)
* [Vdeo](https://github.com/marcomusy/vedo)
    
## Geospatial General
* [Python resources for earth science](https://github.com/javedali99/python-resources-for-earth-sciences)
* [geoutils](https://github.com/GlacioHack/geoutils) -> geospatial analysis and foster inter-operability between other Python GIS packages.
## Vector Data
### Python
* [Geopandas](https://geopandas.org/en/stable/)
  * [Dask-geopandas](https://github.com/RichardScottOZ/dask-geopandas)
    * [Tutorial](https://github.com/martinfleis/dask-geopandas-tutorial)
  * [geofileops](https://github.com/geofileops/geofileops) -> Increased speed spatial joins via database functions and geopackage
* [Kart](https://github.com/koordinates/kart) -> Distributed version control for daata
* [PyESRIDump](https://github.com/RichardScottOZ/pyesridump) -> Library to grab data at scale from ESRI Rest 
* [DuckDB Spatial Extension](https://duckdb.org/docs/stable/core_extensions/spatial/overview.html) -> Can also use standalone no install necessary executable
### R
* [SF](https://r-spatial.github.io/sf/)
* [terra](https://github.com/rspatial/terra) -> terra provides methods to manipulate geographic (spatial) data in "raster" and "vector" form.  

## Raster Data
### C
* [exactextract](https://github.com/isciences/exactextract) -> command line zonal stats in C
### Julia
* [Rasters.jl](https://github.com/rafaqz/Rasters.jl) -> reading and writing common raster data types
### Python 
* [Rasterio](https://github.com/rasterio/rasterio) -> python base library for raster data handling
 * [georeader](https://github.com/giswqs/georeader) -> process raster data from different satellite missions
* [Rasterstats](https://github.com/perrygeo/python-rasterstats) -> summarising geospatial raster datasets based on vector geometries
* [Xarray](https://github.com/pydata/xarray) -> Multidimensional Labelled array handling and analysis
  * [Rioxarray](https://corteva.github.io/rioxarray/stable/) -> Fabulous high level api for xarray handling of raster data
  * [Geocube](https://github.com/corteva/geocube) -> Rasterisation of vector data api
  * [ODC-GEO](https://github.com/opendatacube/odc-geo/) -> Tools for remote sensing based raster handling with many extremely handy tools like colorisation, grid workflows
  * [Rasterix](https://github.com/dcherian/rasterix) -> Raster tricks for xarray
  * [COG Validator](https://github.com/rouault/cog_validator) -> checking format of cloud optimised geotiffs
  * [Griffine](https://github.com/jkeifer/griffine) -> utilities for working with affine grids
  * [ouroboros](https://github.com/corbel-spatial/ouroboros) -> Extract terrible GDB rasters to something else!
  * [serverless-datacube-demo](https://github.com/earth-mover/serverless-datacube-demo) -> xarray via lithops / Coiled / Modal
  * [tifviewer](https://github.com/nkeikon/tifviewer) -> Lightweight cli geotiff viewer
  * [Xarray Spatial](https://github.com/RichardScottOZ/xarray-spatial) -> Statistical analysis of raster data such as classification like natural breaks
  * [xarray-einstats](https://github.com/arviz-devs/xarray-einstats) -> Stats, linear algebra and einops for xarray
  * [xdggs](https://github.com/RichardScottOZ/xdggs) -> Other types of grids
  * [xgcm](https://github.com/xgcm/xhistogram) -> Histograms with labels
  * [xrft](https://github.com/RichardScottOZ/xrft) -> Xarray based Fourier Transforms
  * [xvec](https://xvec.readthedocs.io/en/stable/index.html) -> Vector data cubes for Xarray
### R  
* [Raster](https://rspatial.org/raster/spatial/8-rastermanip.html) -> R library
* [terra](https://github.com/rspatial/terra) -> provides methods to manipulate geographic (spatial) data in "raster" and "vector" form.
* [stars](https://github.com/r-spatial/stars) -> spatiotemporal Arrays: Raster and Vector Datacubes
* [exactextracr](https://github.com/isciences/exactextractr) -> raster zonal statistics for R
### Benchmarks
* [raster-benchmark](https://github.com/kadyb/raster-benchmark) -> Benchmarking some raster libaries in python and R
#### Gui
* [Whitebox Tools](https://github.com/jblindsay/whitebox-tools) -> python api, gui, etc. have used for topographical wetness index calculation
#### Other
* [Smoothify](https://github.com/DPIRD-DMA/Smoothify) -> Smoothing out polygonised raster data


## Data Collection
* [PiAutoStage](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021GC009693) -> 'An Open-Source 3D Printed Tool for the Automatic Collection of High-Resolution Microscope Imagery;' designed for mineral samples.

## Data Conversion
* [AEM to seg-y](https://github.com/Neil-Symington/AEM2SEG-Y)
* [ASEG GDF2](https://github.com/kinverarity1/aseg_gdf2)
* [CGG Outfile reader](https://github.com/RichardScottOZ/CGG-Out-Reader)
* [Geosoft Grid to Raster](https://github.com/RichardScottOZ/Geosoft-Grid-to-Raster)
* [Loop Geosoft Grid](https://github.com/Loop3D/geosoft_grid)
* [Harmonica Geosoft Grid](https://github.com/fatiando/harmonica/pull/348) -> Pull request in progress on conversion to xarray
* [AuScope](https://github.com/RichardScottOZ/geomodel-2-3dweb) -> Data from binary GOCAD models
* [GOCAD SG Grid Reader](https://github.com/RichardScottOZ/GOCAD_SG_Grid_Reader)
	* [geomodel-2-3dweb](https://github.com/RichardScottOZ/geomodel-2-3dweb) -> In here they have a method to extract data from binary GOCAD SG Grids
* [Leapfrog Mesh Reader](https://github.com/ThomasMGeo/leapfrogmshreader)
* [OMF](https://github.com/gmggroup/omf) -> Open Mining Format for conversion between things
* [PDF Miner](https://github.com/euske/pdfminer)
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
  [geo-lm](https://github.com/williamjsdavis/geo-lm) -> GemPy models via Llama-4
* [Gemgis](https://github.com/cgre-aachen/gemgis) -> Geospatial Data Analysis assistance 
* [LoopStructural](https://github.com/Loop3D/LoopStructural) -> Implicity Modelling
* [Manual python geologia](https://github.com/kevinalexandr19/manual-python-geologia) -> Analysis of geology data
* [Map2Loop](https://github.com/Loop3D/map2loop-2) -> 3D Modelling Automation
  * [Loop3D](https://github.com/Loop3D/Loop3D) -> GUI for Map2Loop
* [Pybedforms](https://github.com/AndrewAnnex/pybedforms)
* [SA Stratigraphy](https://github.com/RADutchie/SA-Strarigraphy-db) -> Stratigraphy database editor webapp
* [Striplog](https://github.com/agile-geoscience/striplog)

* [Analise_de_Dados_Estruturais_Altamira](https://github.com/fnaghetini/Analise_de_Dados_Estruturais_Altamira/blob/main/Analise_de_Dados_Estruturais_Altamira.ipynb)
* [Global Tectonics](https://github.com/dhasterok/global_tectonics) -> Open source dataset to build on, plates, margins etc.
	* [paper](https://www.sciencedirect.com/science/article/abs/pii/S0012825222001532?fr=RR-2&ref=pdf_download&rr=82b54d6f3bca5721)
 * [zenodo additions](https://zenodo.org/record/6586972)
* [Litholog](https://github.com/rgmyr/litholog)
* [pyGPlates](https://www.gplates.org/docs/pygplates/index.html)
 * [Tutorial data](https://www.earthbyte.org/webdav/ftp/earthbyte/GPlates/TutorialData_GPlates2.2.zip)
	* [paper](https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.185)
  * [GplatesReconstructionModel](https://github.com/siwill22/GPlatesReconstructionModel)
* [Truth Tables for Consistency Checking Geological Models](https://zenodo.org/records/13948382)
  * [paper](https://gmd.copernicus.org/articles/18/71/2025/)
 
## Geophysics
* [Geoscience Australia Utilities](https://github.com/RichardScottOZ/geophys_utils)
* [Geophysics for Practicing Geoscientists](https://github.com/geoscixyz/gpg)
* [Potential Field Toolbox](https://github.com/RichardScottOZ/PFToolbox) -> Some xarray based Fast Fourier Transform filters - derivatives, pseudogravity, rpg etc.
  * [Notebook](https://github.com/RichardScottOZ/PFToolbox/blob/master/FFT_Filter.ipynb) -> Class with some examples [Vertical derivative, Pseudogravity, Upward Continuation etc)
 * [Computation geophysics sandbox](https://github.com/yohanesnuwara/computational-geophysics)
 * [RIS Basement Sediment](https://github.com/mdtanker/RIS_basement_sediment) -> Depth to Magnetic Basement in Antarctica
 * [Signal Image Processing](https://github.com/PyBrown/Signal-Image-Processing)
### Electromagnetic
* [Geoscience Australia AEM](https://github.com/GeoscienceAustralia/ga-aem)
* [UH Electromagnetics](https://github.com/jiajiasun/UHElectromagnetics) -> Coursework notebooks on understanding this domain
* [AEM Interpretation](https://github.com/Neil-Symington/aem_interp_dash)
* [EMag Py](https://gitlab.com/hkex/emagpy/-/tree/master) -> FDEM 
* [ResIPy](https://github.com/hkexgroup/resipy) -> DC / IP 
### Gravity and Magnetics 
* [Harmonica](https://github.com/fatiando/harmonica)
 * [Filter examples](https://www.fatiando.org/harmonica/latest/user_guide/transformations.html) -> Fast Fourier transform based processing via xarray
* [Australian Gravity Data](https://github.com/compgeolab/australia-gravity-data)
* [Worms](https://bitbucket.org/fghorow/bsdwormer)
 * [Worms update](https://bitbucket.org/RichardScottOZ/bsdwormer/) <- potential fields worm creation with some minor updates to handle new networkx api
  *[github mirror](https://github.com/RichardScottOZ/BSDWormer)   
* [Osborne Magnetic](https://github.com/fatiando-data/osborne-magnetic) -> Survey data processing example
### Seismic
* [Segyio](https://github.com/equinor/segyio)
* [Segysak](https://github.com/trhallam/segysak) -> Xarray based seg-y data handling and analysis
* [Geophysical notes](https://github.com/aadm/geophysical_notes) -> Seismic data processing
### Magnetotellurics
* [MtPy](https://github.com/RichardScottOZ/mtpy)
 * [Tutorials](https://github.com/simpeg-research/iris-mt-course-2022)
* [MtPy](https://github.com/MTgeophysics/mtpy-v2) -> update of the above to make things easier
* [Mineral Stats Toolkit](https://github.com/RichardScottOZ/mineral-stats-toolkit) -> Distance to MT features analysis
 * [Lithospheric conductors paper](https://www.researchgate.net/publication/360660467_Lithospheric_conductors_reveal_source_regions_of_convergent_margin_mineral_systems)
* [mtwaffle](https://github.com/kinverarity1/mtwaffle) -> MT data analysis examples
* [pyMT](https://github.com/eroots/pyMT)
* [resistics](https://github.com/resistics/resistics)
* [MECMUS](https://github.com/GoFEM/MECMUS) -> tools to read Electrical Conductivity model of the USA
 * [model](https://ds.iris.edu/ds/products/emc-mecmus-2022/)
   
### Gridding
* [GMT](https://github.com/GenericMappingTools/gmt)
  * [PyGMT](https://www.pygmt.org/latest/)
* [Verde](https://github.com/fatiando/verde)
* [Grid_aeromag](https://github.com/rmorel/grid-aeromag) -> Brazilian gridding example
* [pyinterp](https://github.com/CNES/pangeo-pyinterp/tree/master) -> Multidimensional gridding via Boost
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
* [Geochemistrypi](https://github.com/ZJUEarthData/geochemistrypi)
	* [paper](https://www.researchgate.net/publication/377746652_Geochemistry_p_Automated_Machine_Learning_Python_Framework_for_Tabular_Data)
  
## Drilling
* [dh2loop](https://github.com/Loop3D/dh2loop) -> Drilling Interval assistance
	* [paper](https://gmd.copernicus.org/articles/14/6711/2021/)
* [drilldown](https://github.com/cardinalgeo/drilldown) -> Drilling visualisation in notebooks via geoh5py -> note desurveying
* [PyGSLib](https://github.com/opengeostat/pygslib) -> Downhole surveying and interval normalising
* [pyborehole](https://github.com/AlexanderJuestel/pyborehole) -> Processing and visualizing borehole data
* [dhcomp](https://github.com/FractalGeoAnalytics/dhcomp/tree/master) -> composites geophysical data to a set of intervals

## Remote Sensing
* [Awesome spectral indices](https://github.com/davemlz/awesome-spectral-indices) -> Guide to spectral index creation
* [Open Data Cube](https://www.opendatacube.org/)
  * [DEA Notebooks](https://github.com/GeoscienceAustralia/dea-notebooks) -> Code for use in ODC style workflows
  * [Datacube-stats](https://github.com/daleroberts/datacube-stats) -> Statistical analysis library for ODC
  * [Geo Notebooks](https://github.com/Element84/geo-notebooks) -> Code examples from Element 84
 * [Raster4ML](https://github.com/remotesensinglab/raster4ml) -> A large number of vegetation indices
 * [Lefa](http://lefa.geologov.net) -> Fracture analysis, lineaments
## Serverless
* [Kerchunk](https://github.com/RichardScottOZ/kerchunk) -> Serverless access to cloud based data via Zarr
	* [Kerchunk geoh5](https://github.com/RichardScottOZ/Kerchunk-geoh5) -> Access to Geoscient Analyst/geoh5 projects serverlessly via kerchunk
* [Tifffile variant](https://github.com/cgohlke/tifffile/issues/125)
* [Virtuallizar](https://github.com/zarr-developers/VirtualiZarr) -> Similar idea to kerchunk
	* [icehunk](https://github.com/earth-mover/icechunk) -> Transactional storage engine for tensor / ND-array data designed for use on cloud object storage. 
### Stac catalogues
* [DEA Stackstac](https://github.com/RichardScottOZ/DEA-stackstac) -> Examples of working with Digital Earth Australia data
* [Intake-stac](https://github.com/intake/intake-stac)
* [ML AOI Extension](https://github.com/stac-extensions/ml-aoi)
* [ML Model Extension Specification](https://github.com/stac-extensions/ml-model) -> Machine Learning Model Specification for CatalogingSpatio-Temporal Models
	* [paper](https://dl.acm.org/doi/10.1145/3681769.3698586)
* [ODC-Stac](https://github.com/opendatacube/odc-stac) -> Database free Open Data Cube
* [Pystac](https://github.com/stac-utils/pystac)
* [Sat-search](https://github.com/sat-utils/sat-search)
* [Stackstac](https://github.com/RichardScottOZ/stackstac) ->  Metadata speeded up dask and xarray timeseries

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
* [Colormap distortions](https://github.com/mycarta/Colormap-distorsions-Panel-app) -> A Panel app to demonstrate distortions created by non-perceptual colormaps on geophysical data
* [Ripping Data from Colormpas](https://gist.github.com/kwinkunks/485190adcf3239341d8bebac94de3a2b)
* [Open Geoscience Code Projects](https://softwareunderground.github.io/open_geosciene_code_projects_viz/explore/)

## Geospatial
* [Geospatial](https://github.com/giswqs/geospatial) >- installs multiple common python packages
* [Geospatial python](https://forrest.nyc/75-geospatial-python-and-spatial-data-science-resources-and-guides/) -> Curated list

## Technology Stacks
### C
* [GDAL](https://github.com/OSGeo/gdal) -> Absolutely crucial data transformation and analysis framework
  * [Tools]() -> Note has many command line tools that are very useful as well
### Julia
* [Julia Earth](https://github.com/JuliaEarth) -> Fostering geospatial data science and geostatistical modeling in Earth sciences  
* [Julia Geodynamics](https://github.com/JuliaGeodynamics) -> computational geodynamics code
* [Introduction to Julia for Geoscience](https://github.com/mauro3/Julia-intro-for-geoscience-EGU2024)
### Python - PyData
* [Anaconda](https://www.anaconda.com/products/distribution) -> Get lots installed already with this package manager.
  * [GDAL et al](https://www.anaconda.com/products/distribution) -> Take the pain out of GDAL and Tensorflow installs here
  * [Git Bash](https://discuss.codecademy.com/t/setting-up-conda-in-git-bash/534473) -> Getting conda to work in Git Bash
* [Numpy Multidimensional arrays](https://numpy.org/)
* [Pandas Tabular data analysis](https://pandas.pydata.org/)
* [Matplotlib visualisation](https://matplotlib.org/)
* [Zarr](https://github.com/zarr-developers/zarr-python) -> Compressed, chunked distributed arrays
* [Dask](https://github.com/dask/dask) -> Parallel, distributed computing
  * [Dask Cloud Provider](https://github.com/RichardScottOZ/dask-cloudprovider) -> Automatically start dask clusters on the cloud
  * [Dask Median](https://gist.github.com/andrewdhicks/d89849997453cdfad6fa568816ca7160) -> Notebook giving a Dask median function prototype
* [Python Geospatial Ecosystem](https://github.com/loicdtx/python-geospatial-ecosystem) -> Curated information
### Rust - GeoRust
* [GeoRust](https://georust.org/) -> Collection of geospatial utilities in rust
### Databases
* [DuckDB](https://github.com/duckdb/duckdb) -> In process OLAP DB at speed - has some geospatial and array capabilities
 * [ibis + Duckdb geopsatial](https://github.com/ncclementi/ibis_duckdb_geospatial_scipy2024) -> scipy2024 talk

## Data Science
* [Python Data Science Template](https://github.com/RichardScottOZ/python-data-science-template) -> Project package setup
* [Awesome python data science](https://github.com/krzjoa/awesome-python-data-science) -> Curated guide

## Probability
* [distfit](https://github.com/erdogant/distfit) -> Probability density fitting

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
* [Geological Society of Western Australia](https://github.com/Geological-Survey-of-Western-Australia/Vocabularies)
* [Stratigraphic](https://github.com/GeoscienceAustralia/strat-ontology-graph-API)
* [Geoscience Knowledge Manager](https://github.com/Loop3D/GKM)
* [GeoSciML Vocabularies](https://geosciml.org/resource/def/voc/)

# Books
## Python
* [Python geospatial analysis cookbook](https://github.com/mdiener21/python-geospatial-analysis-cookbook)
* [Geoprocessing with Python](https://livebook.manning.com/book/geoprocessing-with-python/about-this-book/31) -> Manning livebook
## Other
* [Textbook](https://github.com/rougier/scientific-visualization-book)
* [Machine Learning in the Oil and Gas industry](https://github.com/Apress/machine-learning-oil-gas-industry)
* [Geocomputation with R](https://github.com/Robinlovelace/geocompr)
* [Earthdata Cloud Cookbook](https://github.com/NASA-Openscapes/earthdata-cloud-cookbook) -> How to access NASA resources
* [Data Cleaner's Cookbook](https://www.datafix.com.au/cookbook/about.html) -> Putting unix tools to good use for data wrangling and cleaning
* [Encyclopedia of Mathematical Geosciences](https://link.springer.com/referencework/10.1007/978-3-030-26050-7?page=1#toc)
* [Handbook of Mathematical Geosciences](https://link.springer.com/book/10.1007/978-3-319-78999-6?page=2#toc)

# Other
* [GXPy](https://github.com/GeosoftInc/gxpy) -> Geosoft Python API
* [EarthArxiv](https://github.com/eartharxiv/API/issues) -> Download papers from the preprint archive
* [Essoar](https://www.essoar.org/) -> Preprint paper archive


# Datasets

## World
### Drilling
#### Ocean
* [Deep Ocean](https://www.ncei.noaa.gov/products/international-ocean-drilling-archive)
#### Oil and Gas Wells
* [GOGI](https://edx.netl.doe.gov/dataset/global-oil-gas-features-database) -> GOGI Oil and Gas wells
  * [report](https://edx.netl.doe.gov/dataset/development-of-an-open-global-oil-and-gas-infrastructure-inventory-and-geodatabase)
* [OGIM](https://zenodo.org/records/15103476) -> Oil and Gas Infrastructure Mapping [including wells]
  * [paper](https://essd.copernicus.org/articles/15/3761/2023/) -> OGIM paper
  * [supplement](https://essd.copernicus.org/articles/15/3761/2023/essd-15-3761-2023-supplement.pdf) -> Permian Basin ML check
    * [paper](https://acp.copernicus.org/articles/21/6605/2021/) -> Wellpad detection from space
### Geochemistry
* [Critical Minerals in Ores](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/145496)
### Geology
* [Bedrock](https://osdp-psdo.canada.ca/dp/en/search/metadata/NRCAN-GEOSCAN-1-2237677) -> Generalised geology of the world
* [GLIM](https://doi.pangaea.de/10.1594/PANGAEA.788537) -> Global lithology map
* [Paleogeology](https://www.annualreviews.org/doi/suppl/10.1146/annurev-earth-081320-064052) An Atlas of Phanerozoic Paleogeographic Maps
* [Sedimentary Layers](https://daac.ornl.gov/SOILS/guides/Global_Soil_Regolith_Sediment.html) -> Global 1-km Gridded Thickness of Soil, Regolith, and Sedimentary Deposit Layers
* [World Stress Map](https://www.world-stress-map.org/) -> Global compilation of information on the crustal present-day stress field
* [GMBA](https://www.gmba.unibe.ch/services/tools/mountain_inventory_v1/index_eng.html) -> Global mountain inventory
### Geophysics
#### Gravity
* [Curvature](https://www.3dearth.uni-kiel.de/en/public-data-products) -> Global curvature analysis from gravity gradient data
* [WGM 2012](https://bgi.obs-mip.fr/data-products/grids-and-models/wgm2012-global-model/)
#### Magnetics
* [EAMG2V3](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ngdc.mgg.geophysical_models:EMAG2_V3) _> Earth Magnetic Anomaly Grid
* [WDMAM](https://geomag.org/models/wdmam.html) -> World Digital Magnetic Anomaly Map
#### Magnetotellurics
* [EMC](https://ds.iris.edu/ds/products/emc-globalem-2015-02x02/) -> global 3D inverse model of electrical conductivity 
#### Seismic
* [LAB SLNAAFSA](https://static-content.springer.com/esm/art%3A10.1038%2Fs41561-020-0593-2/MediaObjects/41561_2020_593_MOESM2_ESM.gz)
* [LAB CAM2016](https://static-content.springer.com/esm/art%3A10.1038%2Fs41561-020-0593-2/MediaObjects/41561_2020_593_MOESM2_ESM.gz)
* [Moho](http://gocedata.como.polimi.it/wcs.php) -> GEMMA Data
* [Moho](https://nextcloud.ifg.uni-kiel.de/index.php/s/PS2owBsPznj5gpb) -> Szwillus Data
* [Seismic Velocity](https://ds.iris.edu/ds/products/emc-dbrd_nature2020/) - > Debayle et al
* [LithoRef18](https://www.juanafonso.com/software) -> A global reference model of the lithosphere and upper mantle from joint inversion and analysis of multiple data sets
* [CRUST1.0](https://ds.iris.edu/ds/products/emc-crust10/) -> global crustal model netcdf
 * [Overview homepage](https://igppweb.ucsd.edu/~gabi/crust1.html)
#### Thermal
* [Heat Flow](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2019GC008389)
#### General
* [Deep Time Digital Earth](https://deep-time.org/data) -> Data and visualisation for a variety of data sources and models
* [EarthChem](https://www.earthchem.org/) -> Community-driven preservation, discovery, access, and visualization of geochemical, geochronological, and petrological data
* [GEOROC](https://data.goettingen-research-online.de/dataverse/digis) -> Geochemical composition of rocks
  * [georoc-data](https://github.com/pofatu/georoc-data?tab=readme-ov-file)
* [global geology](https://github.com/siwill22/global-geology) -> A short recipe to make a global geology map in GIS format (e.g. shapefile), with age ranges mapped to the GTS2020 timescale
* [Large Igenous Provinces Commission](http://www.largeigneousprovinces.org/links)
* [Mantle Plumes](http://www.mantleplumes.org/SLIPs.html)
* [Sediment Thickness](https://igppweb.ucsd.edu/%7Egabi/sediment.html) -> Map
* [spatialreference.org](https://github.com/neteler/spatialreference.org) -> repository for the website

## Australia
* [Common Earth Model](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/101380)
* [Heavy Mineral Map](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/148916)
 * [Heavy MIneral Map of Australia Pilot](https://www.mdpi.com/2075-163X/12/8/961)
 * [Shiny App](https://geoscienceaustralia.shinyapps.io/mna4hm/)
### Geochemistry
* [Predictive grids of major oxide concentrations in surface rock and regolith over the Australian continent](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/148587) -> Various oxides
### Geology
* [Alkaline Rocks Atlas](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147963)
 * [Cenozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/4ef3b60d-d7ee-4810-af66-b0948dae4acb)
 * [Mesozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/7d3ace8d-9771-4090-926d-d618db5071bf)
 * [Paleozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/63a29e5e-041e-42ad-a540-8125aa442624)
 * [Archaean](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/7120ce29-606f-4329-abbf-dd6cb56a1cbf)
 * [search](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/search?keyword=Alkaline%20rocks)
 * [Proterozoic Alkaline Rocks](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147893) -> Proterozoic alkaline and related igneous rocks of Australia GIS
  * [Cenozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/a98dc23a-9666-4568-807c-09910646603a)
  * [Mesozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/a8d7e179-9a03-4c4e-bcae-e224917f8f2a)
  * [Paleozoic](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/5b490317-7cfc-45de-8597-003a8f89dddd)
  * [Archaean](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/7c0a177d-9733-452c-95e2-671af856e54b)
	* [paper](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147894)
 https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147963
* [Hydrogeology](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/32368) -> Hydrogeology Map of Australia
 * [Hydrogeology](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/15629) -> 5M
* [Layered Geology](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/149179) -> 1M 
* [Surface Geology](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/74855) -> 1M Scale
* [The Australian Mafic-Ultramafic Magmatic Events GIS Dataset](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/82166)
### Geophysics
* [Gravity](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/133023) -> 2019 Australian National Gravity Grids
#### Magnetics
* [TMI](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/144733) -> Magnetic Anomaly Map of Australia, Seventh Edition, 2019 TMI
 * [40m](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131512) -> 40m version
* [VRTP](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131519) -> Total Magnetic Intensity (TMI) Grid of Australia with Variable Reduction to Pole (VRTP) 2019 
* [1VD](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/132275) -> Total Magnetic Intensity Grid of Australia 2019 - First Vertical Derivative (1VD)
#### Radiometrics
* [Radiometrics](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/144413) -> Complete Radiometric Grid of Australia (Radmap) v4 2019 with modelled infill
* [K](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131962) -> Radiometric Grid of Australia (Radmap) v4 2019 filtered pct potassium grid
* [U](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131974) -> Radiometric Grid of Australia (Radmap) v4 2019 filtered ppm uranium 
* [Th](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131967) -> Radiometric Grid of Australia (Radmap) v4 2019 filtered ppm thorium
* [Th/K](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131976) -> Radiometric Grid of Australia (Radmap) v4 2019 ratio thorium over potassium
* [U/K](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131964) -> Radiometric Grid of Australia (Radmap) v4 2019 ratio uranium over potassium
* [U/Th](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131983) -> Radiometric Grid of Australia (Radmap) v4 2019 ratio uranium over thorium
* [U squared/Th](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131981) -> Radiometric Grid of Australia (Radmap) v4 2019 ratio uranium squared over thorium
* [Dose Rate](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131960)-> Radiometric Grid of Australia (Radmap) v4 2019 filtered terrestrial dose rate
* [Ternary Picture](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/134857) -> Radiometric grid of Australia (Radmap) v4 2019 - Ternary image (K, Th, U)
#### AusAEM
* [AusAEM 1](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/132709) -> AusAEM Year 1 NT/QLD Airborne Electromagnetic Survey; GA Layered Earth Inversion Products
* [AusAEM 1](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/124092) -> AusAEM Year 1 NT/QLD: TEMPEST® airborne electromagnetic data and Em Flow® conductivity estimates
* [AusAEM 1](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/145120) -> AusAEM1 Interpretation Data Package
* [AusAEM 2](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/140156) -> AusAEM 02 WA/NT 2019-20 Airborne Electromagnetic Survey
* [AusAEM–WA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146345) -> AusAEM–WA, Murchison Airborne Electromagnetic Survey Blocks
* [AusAEM–WA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146042) -> AusAEM-WA, Southwest-Albany Airborne Electromagnetic Survey Blocks
* [AusAEM–WA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/144621) -> AusAEM WA 2020-21, Eastern Goldfields & East Yilgarn Airborne
* [AusAEM–WA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/145265) -> AusAEM (WA) 2020-21, Earaheedy & Desert Strip
* [AusAEM ERC](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/145744) -> AusAEM Eastern Resources Corridor
* [AusAEM WRC](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147688) -> AusAEM Western Resources Corridor
 * [interp overview](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147251)
* [National surface and near-surface conductivity grids](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/148588) -> National ML interpolation for AusEM in similar fashion to Northern Australia
#### AusLAMP
* [AusLAMP SEA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/131889) -> Resistivity model of the southeast Australian mainland from AusLAMP magnetotelluric data
 * [Victoria Data](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/120864)
 * [NSW Data](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/132148)
* [AusLAMP TISA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/130832) -> Resistivity model derived from magnetotellurics: AusLAMP-TISA project 
* [AusLAMP Delamerian](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/144077) -> Lithospheric resistivity model of the Delamerian Orogen from AusLAMP magnetotelluric data
* [AusLAMP NE SA](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/1cab41af-eddf-40e4-b091-a6e733e5701a)
* [AusLAMP Gawler](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/1cab41af-eddf-40e4-b091-a6e733e5701a)
 * [AusLAMP Stations](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/116721) -> circa 2017
 * [Tasmanides Paper](https://www.sciencedirect.com/science/article/pii/S0040195120302432?via%3Dihub)
#### Moho
* [Moho 2019 and 2023](https://auspass.edu.au/research/)
	* [paper](https://academic.oup.com/gji/article/233/3/1863/7008903?login=false)
#### Mineral Deposits
* [Geological setting, age and endowment of major Australian mineral deposits](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/145716)
* [A Comprehensive dataset for Australian mine production 1799 to 2021](https://doi.org/10.25439/rmt.22724081.v2)
	* [paper](https://www.nature.com/articles/s41597-023-02275-z#Sec3)
#### Mineral Potential
* [Overview - Geoscience Australia](https://www.eftf.ga.gov.au/mineral-potential-mapping) -> Overview of publications and datasets
* [Sediment Hosted Zinc](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147425)
 * [Report](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147540)
* [Sediment Hosted Copper](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147425)
 * [Report](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147539)
 * [Abstract](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147540)
* [Carbonatite Rare Earth Elements](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147865)
	* [paper](https://www.researchgate.net/publication/373752600_A_national-scale_mineral_potential_assessment_for_carbonatite-related_rare_earth_element_mineral_systems_in_Australia)
#### Mine Waste
* [Australian Mine Waste](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/147516)
#### Native Title
* [National Native Title Tribunal](https://data-nntt.opendata.arcgis.com/maps/2698667a86e54550b732174a71c3bc57/about)
 
### Remote Sensing
* [Landsat Bare Earth](http://dea-public-data.s3-website-ap-southeast-2.amazonaws.com/?prefix=geomedian-australia/) - Bare earth median from Landsat
 * [Enhanced barest earth Landsat imagery for soil and lithological modelling: Dataset](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/144231) -> Details of an enhancement
* [Global mining footprint mapped from high-resolution satellite imagery](https://zenodo.org/record/6806817#.ZEmpinZBxD8)
** [Paper](https://www.nature.com/articles/s43247-023-00805-6)
* [DEM](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/72759) -> Australia 1 sec SRTM DEM of various varieties
### Structure
* [Major Crustal Boundaries of Australia - 2024 Edition](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/149663)
### Velocity
* [AU Tomo](https://data.csiro.au/collection/csiro:51008) -> Next-generation velocity model of the Australian crust from synchronous and asynchronous ambient noise imaging
### Topography
* [Multiscale Topographic Position](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/123119) - RGB
 * [Info](https://portal.tern.org.au/metadata/102.100.100/9241)
* [Topographic Wetness Index](https://data.csiro.au/collection/csiro:5588v2) - 1 and 3 arc seconds
 * [Info](https://portal.tern.org.au/metadata/102.100.100/9241)
* [Topographic Position Index](https://data.csiro.au/collection/csiro:5144) - 1 and 3 arc seconds
 * [Info](https://portal.tern.org.au/metadata/102.100.100/8339)
* [Weathering Intensity Model](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/123106)
	* [paper](https://www.sciencedirect.com/science/article/abs/pii/S0016706111000036)
 * [Info](https://www.ga.gov.au/ausgeonews/ausgeonews201103/weathering.jsp_)
 * {Info](https://researchdata.edu.au/weathering-intensity-model-australia/1361069)
### Northern
* [Cover thickness TISA](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/130734) -> Cover thickness points for Tennant Creek Mt Isa with interpolated grids
* [High resolution conductivity mapping using regional AEM survey and machine learning](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146163) -> ML conductivity interpolation for AusAEM
 * [Extended abstract](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/146380)
* [Solid Geology](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/135277) -> Solid Geology of the North Australian Craton
* [Inversion Models](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/145901) -> The North Australian Craton 3D Gravity and Magnetic Inversion Models
* [Ni-Cu-PGE](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/83884) -> Potential for intrusion-hosted Ni-Cu-PGE sulfide deposits in Australia: A continental-scale analysis of mineral system prospectivity
* [TISA IOCG](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/130587) -> Iron oxide copper-gold (IOCG) mineral potential assessment for the Tennant Creek – Mt Isa region: geospatial data
* [TISA Alteration](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/123013) -> Producing Magnetite and Hematite Alteration Proxies using 3D Gravity and Magnetic Inversion

### South Australia
#### Geology
* [Bedrock Geology](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/103461)
* [Crystalline Basement](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/216146e3-7cb0-410a-bcb2-68e5e6bcf9f4) -> Crystalline basement intersecting drillholes
* [Mines and Mineral Deposits](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/a0e4b62c-ec88-44b8-a530-b4e744a6b414)
* [Mineral Drillholes](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/3e6692fd-7461-494c-b40b-4cd8738fc762)
* [Solid Geology 3D](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/a3a22f6a-afd0-4d97-9f71-7780ab17e5f9)
 * [100K Faults](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/50b5a79c-9764-414e-b28d-548dbe006cb3)
 * [Archaean](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/79d50974-a57e-4754-91aa-1f0be6489a8c)
 * [Archaean Faults](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/8e517d3f-8255-4159-8bcf-3e101622a2c2)
 * [Mesoproterozoic](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/3e8bef60-ff90-44ea-a246-cdbf94e834c3) -> Middle
 * [Mesoproterozoic](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/0bed7f88-5ff3-4cd3-8c7a-6c733b9d47fc) -> Middle faults
 * [Mesoproterozoic](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/89283327-86c9-4128-a8ad-4da63668b136) - > Late
 * [Mesoproterozoic Faults](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/a4e4313f-f932-4e06-9399-c6a52eb9bfcb) -> Late faults
 * [Neoproterozoic](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/267280d7-8d57-498c-a9ee-ab1f1e52d696)
 * [Neoproterozoic faults](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/7e5ff842-36cf-4b91-87c2-1e715ef62f0d)
* [Stuart Shelf Sedimentary Copper 3D Model](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/ee68f692-8f5c-4b7f-a842-a9d381bf6f26)
* [Surface Geology](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/e27f9a25-b749-4dba-bfb3-ca90baf04d79)
#### Geophysics
* [AusLAMP 3D](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/1cab41af-eddf-40e4-b091-a6e733e5701a) -> Magnetotelluric inversions
* [GCAS](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/1ce5ea8f-b6f6-42bb-9061-6a95862840d5) -> Gawler Craton Airborne Survey
* [Gravity](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/f1c5ab7f-21c8-4ed3-85fa-14f6dba40ca6) -> Gravity grids
 * [Stations](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/c6c2c74e-3567-4dbd-8261-c08cda3969c0) -> Gravity stations
* [Magnetics](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/1a797550-8d6b-4ab1-81f2-be6ed15f06d0) -> Magnetics
* [Seismic Lines](https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/397d3203-5d96-4c61-9ac9-1d936d66293f) -> Seismic lines
##### Gawler
* [Gawler MPP](https://webarchive.nla.gov.au/awa/20160615192753/http://www.ga.gov.au/cedda/data/760) -> Gawler Mineral Promotion Project - Data
### Queensland
* [Overview](https://smi.uq.edu.au/files/80453/SREP_NDP_summary_Dec2021.pdf)
* [Deep Mining Queensland](https://smi.uq.edu.au/project/deep-mining-queensland)-> Deep Mining Queensland
* [Deposit Atlas](https://smi.uq.edu.au/project/nw-mineral-province-deposit-atlas) -> Northwest Mineral Province Deposit Atlas
* [Geology](https://qldspatial.information.qld.gov.au/catalogue/custom/detail.page?fid={0CEA14DE-D170-4CA2-9FC9-20ED49143B4B}) -> Geology series overview
* [Mineral and Energy Report](https://geoscience.data.qld.gov.au/data/report/cr102061) -> NORTH-WEST QUEENSLAND MINERAL AND ENERGY PROVINCE REPORT 2011 - NWQMEP
* [Vectoring](https://geoscience.data.qld.gov.au/data/report/cr126164) -> Mineral geochemistry vectoring
* [Petroleum Wells](https://qldspatial.information.qld.gov.au/catalogue/custom/detail.page?fid={CBBE665F-60A8-4116-87C8-AEBF0D21B97C})
* [Coal Seam Gas Wells](https://qldspatial.information.qld.gov.au/catalogue/custom/detail.page?fid={C45038EB-BB83-4B16-9231-1905ED753D77})
* [Drillholes](https://qldspatial.information.qld.gov.au/catalogue/custom/detail.page?fid={9ED7F9ED-456A-4D87-AD30-69231A6F5811})
#### Cloncurry
* [Toolkit](https://geoscience.data.qld.gov.au/data/report/cr126168) -> Multielement toolkit and laboratory
### Northern Territory
* [Arunta IOCG](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/76423) -> Iron oxide-copper-gold potential of the southern Arunta Region
* [South Uranium](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/74288) -> Southern Northern Territory uranium and geothermal energy systems assessment digil data package
* [Tennant Creek](https://ecat.ga.gov.au/geonetwork/srv/eng/catalog.search#/metadata/135011) -> Conductivity Model Derived from Magnetotelluric Data in the East Tennant Region, Northern Territory
### New South Wales
#### Geology
* [Seamless Geology](https://search.geoscience.nsw.gov.au/product/9232) -> NSW Seamless Geology Data Package (older version also on this page)
#### Mineral Potential Data Packages
* [Curnamona](https://search.geoscience.nsw.gov.au/product/9233/7564550)
* [Eastern Lachlan](https://search.geoscience.nsw.gov.au/product/9253)
* [Central Lachlan](https://search.geoscience.nsw.gov.au/product/9261/7922136)
* [Southern New England](https://search.geoscience.nsw.gov.au/product/9222/7371775)
### Western Australia
#### Geochemistry
* [Mafic/Ultramafic](https://data.csiro.au/collection/csiro:58059)
#### Geology
* [100K Bedrock](https://dasc.dmirs.wa.gov.au/Download/File/1773)
 * 100K mapsheets for surface you have to download individually and combine - they aren't consistent
 * 250K mapsheets for surface you have to download individually and combine - they aren't consistent
* [500K Bedrock](https://catalogue.data.wa.gov.au/dataset/1-500-000-state-interpreted-bedrock-geology-dmirs-016)
* [Abandoned Mines](https://catalogue.data.wa.gov.au/dataset/abandoned-mines)
* [Mineral Occurrences](https://catalogue.data.wa.gov.au/dataset/minedex-dmirs-001)
#### Mineral Potential
* [Komatiite-hosted nickel](http://www.dmp.wa.gov.au/msa/komatiite-ni-489.aspx)
 * [Report](https://dmpbookshop.eruditetechnologies.com.au/product/komatiite-hosted-ni-cu-pge-deposits-a-mineral-systems-analysis.do)
#### Prospectivity
* [Capricorn](https://data.csiro.au/collection/csiro:36617v1)-> Prospectivity analysis using a mineral systems approach - Capricorn case study project
* [King Leopold](https://dmpbookshop.eruditetechnologies.com.au/product/mineral-prospectivity-of-the-king-leopold-orogen-and-lennard-shelf-analysis-of-potential-field-data-in-the-west-kimberley-region-geographical-product-n14bnzp.do) -> Mineral prospectivity of the King Leopold Orogen and Lennard Shelf: analysis of potential field data in the west Kimberley region
* [Yilgarn Gold](https://d28rz98at9flks.cloudfront.net/82617/Y4_Gold_Targeting.zip)
* [Yilgarn 2](https://researchdata.edu.au/predictive-mineral-discovery-mineral-system/1209568) -> Predictive mineral discovery in the eastern Yilgarn Craton: an example of district-scale targeting of an orogenic gold mineral system
* [Shop note] -> WA has a few prospectivity packages available to purchase on USB drive for 50-60AU type prices - see in geospaital maps section
### Tasmania
#### Geology
* [250k](https://www.mrt.tas.gov.au/products/digital_data/1250,000_geology_data_download)
* [500k](https://www.mrt.tas.gov.au/products/digital_data/1500,000_geology_data_download)
* [25K](https://www.mrt.tas.gov.au/products/digital_data/125,000_geology_data_download)
* [Mineral Occurrences](https://www.mrt.tas.gov.au/products/digital_data/state_deposits_data_download)
* [3D Model](https://www.mrt.tas.gov.au/products/database_searches/3d_model_data)
### Victoria
* [Seamless Geology](https://www.mrt.tas.gov.au/products/digital_data/state_deposits_data_download)
## New Zealand
* [Mineral Data Pack](https://www.nzpam.govt.nz/maps-geoscience/minerals-datapack/) -> Mineral Exploration Data Pack

## North Americia
* [National-Scale Geophysical, Geologic, and Mineral Resource Data and Grids](https://www.sciencebase.gov/catalog/item/6193e9f3d34eb622f68f13a5) -> Also has some Australia data
* [Groundwater wells](https://www.hydroshare.org/resource/8b02895f02c14dd1a749bcc5584a5c55/) -> Database
* [Maximum horizontal stress orientation and relative stress magnitude (faulting regime) data throughout North America](https://www.sciencebase.gov/catalog/item/6469516bd34e3a6027e2f527) 
## Canada
### Geochemistry
* [Database of Canadian surveys](https://geochem.nrcan.gc.ca/cdogs/content/main/home_en.htm) -> some standardisation done here
### Geology
* [Map](https://geoscan.nrcan.gc.ca/starweb/geoscan/servlet.starweb?path=geoscan/downloade.web&search1=R=208175)
* [Geology](https://ostrnrcan-dostrncan.canada.ca/entities/publication/8ac30d9e-5be6-44b6-8119-848ab893f1e7) -> Updated Bedrock geology map
 * [Geology](https://geoscan.nrcan.gc.ca/starweb/geoscan/servlet.starweb?path=geoscan/downloade.web&search1=R=292232) -> Bedrock geology compilation and regional synthesis of south Rae and parts of Hearne domains, Churchill Province, Northwest Territories, Saskatchewan, Nunavut, Manitoba and Alberta
* [Moho](https://geoscan.nrcan.gc.ca/starweb/geoscan/servlet.starweb?path=geoscan/downloade.web&search1=R=305396) -> National database of Moho depth estimates estimates from seismic refraction and teleseismic surveys
### Geophysics
* [Dap Search](http://gdr.agg.nrcan.gc.ca/gdrdap/dap/search-eng.php) -> Geoportal search - note annoyingly these are in Geosoft grids - see elsewere for conversion possibilties
 * [Gravity, Magnetics, Radiometrics] -> Mostly country scale
 
## Europe 
### European 
* [Dataset for Critical Minerals Hard Rock Deposits](https://zenodo.org/records/15234833)
### Finland
* [FODD](https://www.gtk.fi/en/fennoscandian-mineral-deposits-application-ore-deposits-database-and-maps/) -> Fennoscandian Mineral Deposits
### Ireland
* [MPM](https://www.gsi.ie/en-ie/data-and-maps/Pages/Minerals.aspx) -> Mineral Potentinal Mapping project

# Papers with Code 
### NLP
- https://www.sciencedirect.com/science/article/pii/S2590197422000064?via%3Dihub#bib20- -> Geoscience language models and their intrinsic evaluation -> NRCan code above [includes model]
- https://www.researchgate.net/publication/334507958_Word_embeddings_for_application_in_geosciences_development_evaluation_and_examples_of_soil-related_concepts -> GeoVec [includes model]
- https://www.researchgate.net/publication/347902344_Portuguese_word_embeddings_for_the_oil_and_gas_industry_Development_and_evaluation -> PetroVec [includes model]
- A resource for automated search and collation of geochemical datasets from journal supplements

### Geochemistry
- https://www.researchgate.net/publication/365758387_A_resource_for_automated_search_and_collation_of_geochemical_datasets_from_journal_supplements
	- https://github.com/erinlmartin/figshare_geoscrape?s=09
### Geology
- https://github.com/sydney-machine-learning/autoencoders_remotesensing -> Stacked Autoencoders for Lithological Mapping
### Mineral
- https://www.researchgate.net/publication/318839364_Network_analysis_of_mineralogical_systems
# Papers with Features Data
- These you can reproduce the output geospatially from the data given.
### Mineral Prospectivity
- https://www.sciencedirect.com/science/article/pii/S016913682100010X#s0135 -> Prospectivity modelling of Canadian magmatic Ni (±Cu ± Co ± PGE) sulphide mineral systems [well worth reading]
- https://www.sciencedirect.com/science/article/pii/S0169136821006612#b0510 -> Data–driven prospectivity modelling of sediment–hosted Zn–Pb mineral systems and their critical raw materials [well worth reading]
- https://www.researchgate.net/publication/358956673_Towards_a_fully_data-driven_prospectivity_mapping_methodology_A_case_study_of_the_Southeastern_Churchill_Province_Quebec_and_Labrador

### England
- https://www.researchgate.net/publication/358083076_Machine_learning_for_geochemical_exploration_classifying_metallogenic_fertility_in_arc_magmas_and_insights_into_porphyry_copper_deposit_formation

### Geochemistry
- https://www.researchgate.net/publication/361076789_Automated_machine_learning_pipeline_for_geochemical_analysis

### Geology
- https://eprints.utas.edu.au/32368/ -> Machine-assisted modelling of lithology and metasomatism

### Geophysics
- https://github.com/TomasNaprstek/Aeromagnetic_CNN - Aeromagnetic CNN
 - Paper https://www.researchgate.net/publication/354772176_Convolution_Neural_Networks_Applied_to_the_Interpretation_of_Lineaments_in_Aeromagnetic_Data
 - [PhD](https://zone.biblio.laurentian.ca/bitstream/10219/3549/1/Naprstek%20PhD%20Thesis%20V4.pdf) -> New Methods for the Interpolation and Interpretation of Lineaments in Aeromagnetic Data
 - Paper https://www.researchgate.net/publication/354772176_Convolution_Neural_Networks_Applied_to_the_Interpretation_of_Lineaments_in_Aeromagnetic_Data -> Convolution Neural Networks Applied to the Interpretation of Lineaments in Aeromagnetic Data

# Geospatial Output - No Code
- https://geoscience.data.qld.gov.au/report/cr113697 -> NWMP Data-Driven Mineral Exploration And Geological Mapping [CSIRO too]

# Journals
- https://www.sciencedirect.com/journal/artificial-intelligence-in-geosciences -> Artificial Intelligence in Geosciences

# Papers
- Generally Not ML, or no Code/Data and sometimes no availability at all
- Eventually will separate out into things that have data packages or not like NSW Zone studies.
- However, if interested in an area you can often georeference a picture if nothing else as a rough guide.
- Generally these are not reproducible - a few like the NSW prospectivity zone studies and NWQMP are with some work.  
- The occasional paper in this section may be listed above

## New to File
### General
- https://www.researchgate.net/publication/337650865_A_combinative_knowledge-driven_integration_method_for_integrating_geophysical_layers_with_geological_and_geochemical_datasets
- https://link.springer.com/article/10.1007/s11053-023-10237-w - A New Generation of Artificial Intelligence Algorithms for Mineral Prospectivity Mapping
- https://www.researchgate.net/publication/235443297_Addressing_challenges_with_exploration_datasets_to_generate_usable_mineral_potential_maps
- https://www.researchgate.net/publication/330077321_An_Improved_Data-Driven_Multiple_Criteria_Decision-Making_Procedure_for_Spatial_Modeling_of_Mineral_Prospectivity_Adaption_of_Prediction-Area_Plot_and_Logistic_Functions
- Artificial intelligence and machine learning to enhance critical mineral deposit discovery -> https://www.sciencedirect.com/science/article/pii/S2772883825000111?via%3Dihub
- Artificial intelligence for mineral exploration: A review and perspectives on future directions from data science -> https://www.sciencedirect.com/science/article/pii/S0012825224002691
- https://www.researchgate.net/project/Bayesian-Machine-Learning-for-Geological-Modeling-and-Geophysical-Segmentation
- https://www.researchgate.net/publication/229714681_Classifiers_for_Modeling_of_Mineral_Potential
- https://www.researchgate.net/publication/352251078_Data_Analysis_Methods_for_Prospectivity_Modelling_as_applied_to_Mineral_Exploration_Targeting_State-of-the-Art_and_Outlook
- https://www.researchgate.net/publication/267927728_Data-Driven_Evidential_Belief_Modeling_of_Mineral_Potential_Using_Few_Prospects_and_Evidence_with_Missing_Values
- https://www.linkedin.com/pulse/deep-learning-meets-downward-continuation-caldera-analytics/?trackingId=Ybkv3ukNI7ygH3irCHZdGw%3D%3D
- https://www.researchgate.net/publication/382560010_DINOv2_Rocks_Geological_Image_Analysis_Classification_Segmentation_and_Interpretability
- https://www.researchgate.net/publication/368489689_Discrimination_of_Pb-Zn_deposit_types_using_sphalerite_geochemistry_New_insights_from_machine_learning_algorithm
- https://link.springer.com/article/10.1007/s11430-024-1309-9 -> Explainable artificial intelligence models for mineral prospectivity mapping
- https://www.researchgate.net/publication/229792860_From_Predictive_Mapping_of_Mineral_Prospectivity_to_Quantitative_Estimation_of_Number_of_Undiscovered_Prospects
- https://www.researchgate.net/publication/339997675_Fully_reversible_neural_networks_for_large-scale_surface_and_sub-surface_characterization_via_remote_sensing
  - [arxiv](https://arxiv.org/abs/2003.07474)
  - [presentation](https://slideslive.com/38926360/fully-reversible-neural-networks-for-largescale-surface-and-subsurface-characterization-via-remote-sensing?locale=en)
  - [conference](https://ai4earthscience.github.io/iclr-2020-workshop/papers/ai4earth24.pdf)
  - [juliaCon](https://slim.gatech.edu/Publications/Public/Conferences/JuliaCon/2021/witte2021JULIACONmedlj/witte2021JULIACONmedlj.pdf)
- https://www.researchgate.net/publication/220164488_Geocomputation_of_mineral_exploration_targets
- https://www.researchgate.net/publication/272494576_Geological_knowledge_discovery_and_minerals_targeting_from_regolith_using_a_machine_learning_approach
- https://www.researchgate.net/publication/280013864_Geometric_average_of_spatial_evidence_data_layers_A_GIS-based_multi-criteria_decision-making_approach_to_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/355467413_Harnessing_the_Power_of_Artificial_Intelligence_and_Machine_Learning_in_Mineral_Exploration-Opportunities_and_Cautionary_Notes
- https://www.researchgate.net/publication/335819474_Importance_of_spatial_predictor_variable_selection_in_machine_learning_applications_-Moving_from_data_reproduction_to_spatial_prediction
- https://www.researchgate.net/publication/337003268_Improved_supervised_classification_of_bedrock_in_areas_of_transported_overburden_Applying_domain_expertise_at_Kerkasha_Eritrea - Gazley/Hood
- https://www.researchgate.net/publication/360660467_Lithospheric_conductors_reveal_source_regions_of_convergent_margin_mineral_systems
- https://api.research-repository.uwa.edu.au/portalfiles/portal/5263287/Lysytsyn_Volodymyr_2015.pdf (PhD thesis) GIS-based epithermal copper prospectivity mapping of the Mt Isa Inlier, Australia: Implications for exploration targeting
- https://www.researchgate.net/publication/374972769_Knowledge_and_technology_transfer_in_and_beyond_mineral_exploration -> Knowledge and technology transfer in and beyond mineral exploration
- https://www.researchgate.net/publication/331946100_Machine_learning_for_data-driven_discovery_in_solid_Earth_geoscience
- https://theses.hal.science/tel-04107211/document - Machine Learning Approaches for Sub-surface Geological Heterogeneous Sources
- https://www.researchgate.net/publication/309715081_Magmato-hydrothermal_space_A_new_metric_for_geochemical_characterisation_of_metallic_ore_deposits - Magmato-hydrothermal space: A new metric for geochemical characterisation of metallic ore deposits
- https://www.researchgate.net/publication/220164234_Mapping_complexity_of_spatial_distribution_of_faults_using_fractal_and_multifractal_models_Vectoring_towards_exploration_targets
- https://www.researchgate.net/publication/220163838_Objective_selection_of_suitable_unit_cell_size_in_data-driven_modeling_of_mineral_prospectivity
- https://www.researchgate.net/publication/273500012_Prediction-area_P-A_plot_and_C-A_fractal_analysis_to_classify_and_evaluate_evidential_maps_for_mineral_prospectivity_modeling
- https://www.researchgate.net/publication/354925136_Soil-sample_geochemistry_normalised_by_class_membership_from_machine-learnt_clusters_of_satellite_and_geophysics_data [Gazley/Hood]
- https://link.springer.com/article/10.1007/s12665-024-11870-1 -> Quantification of the uncertainty of geoscientific maps relying on human sensory engagement
- https://www.researchgate.net/publication/235443294_The_effect_of_map-scale_on_geological_complexity
- https://www.researchgate.net/publication/235443305_The_effect_of_map_scale_on_geological_complexity_for_computer-aided_exploration_targeting
- https://link.springer.com/article/10.1007/s11053-024-10322-8 -> Workflow-Induced Uncertainty in Data-Driven Mineral Prospectivity Mapping


### Mineral Prospectivity
#### Australia
- https://www.mdpi.com/2072-4292/15/16/4074 -> A Spatial Data-Driven Approach for Mineral Prospectivity Mapping
- https://www.researchgate.net/publication/353253570_A_Truly_Spatial_Random_Forests_Algorithm_for_Geoscience_Data_Analysis_and_Modelling
- https://www.researchgate.net/publication/253217016_Advanced_methodologies_for_the_analysis_of_databases_of_mineral_deposits_and_major_faults
- https://www.researchgate.net/publication/362260616_Assessing_the_impact_of_conceptual_mineral_systems_uncertainty_on_prospectivity_predictions
- https://www.researchgate.net/publication/352310314_Central_Lachlan_Mineral_Potential_Study
- https://meg.resourcesregulator.nsw.gov.au/sites/default/files/2024-05/EITH%202024%20Muller_Exploration_in_the_House_keynote.pdf -> Critical minerals – prospectivity mapping using generative AI
- https://www.tandfonline.com/doi/pdf/10.1080/22020586.2019.12073159?needAccess=true - > Integrating a Minerals Systems Approach with Machine Learning: A Case Study of ‘Modern Minerals Exploration’ in the Mt Woods Inlier – northern Gawler Craton, South Australia
- https://www.researchgate.net/publication/365697240_Mineral_potential_modelling_of_orogenic_gold_systems_in_the_Granites-Tanami_Orogen_Northern_Territory_Australia_A_multi-technique_approach
- https://publications.csiro.au/publications/publication/PIcsiro:EP2022-0483 -> Signatures of Key Mineral Systems in the Eastern Mount Isa Province, Queensland: New Perspectives from Data Analytics
- https://link.springer.com/article/10.1007/s11004-021-09989-z -> Stochastic Modelling of Mineral Exploration Targets
- https://www.researchgate.net/publication/276171631_Supervised_Neural_Network_Targeting_and_Classification_Analysis_of_Airborne_EM_Magnetic_and_Gamma-ray_Spectrometry_Data_for_Mineral_Exploration
- https://www.researchgate.net/publication/353058758_Using_Machine_Learning_to_Map_Western_Australian_Landscapes_for_Mineral_Exploration 
- https://www.researchgate.net/publication/264535019_Weights-of-evidence_and_logistic_regression_modeling_of_magmatic_nickel_sulfide_prospectivity_in_the_Yilgarn_Craton_Western_Australia
#### Argentina
- https://www.researchgate.net/publication/263542691_ANALYSIS_OF_SPATIAL_DISTRIBUTION_OF_EPITHERMAL_GOLD_DEPOSITS_IN_THE_DESEADO_MASSIF_SANTA_CRUZ_PROVINCE
- https://www.researchgate.net/publication/263542560_EVIDENTIAL_BELIEF_MAPPING_OF_EPITHERMAL_GOLD_POTENTIAL_IN_THE_DESEADO_MASSIF_SANTA_CRUZ_PROVINCE_ARGENTINA
- https://www.researchgate.net/publication/277940917_Porphyry_epithermal_and_orogenic_gold_prospectivity_of_Argentina
- https://www.researchgate.net/publication/269518805_Prospectivity_for_epithermal_gold-silver_deposits_in_the_Deseado_Massif_Argentina
- https://www.researchgate.net/publication/235443303_Prospectivity_mapping_for_multi-stage_epithermal_gold_mineralization_in_Argentina
#### Brazil
- https://www.researchgate.net/publication/367245252_Geochemical_multifractal_modeling_of_soil_and_stream_sediment_data_applied_to_gold_prospectivity_mapping_of_the_Pitangui_Greenstone_Belt_northwest_of_Quadrilatero_Ferrifero_Brazil
- https://www.researchgate.net/publication/381880769_How_do_non-deposit_sites_influence_the_performance_of_machine_learning-based_gold_prospectivity_mapping_A_study_case_in_the_Pitangui_Greenstone_Belt_Brazil
- https://www.researchsquare.com/article/rs-5066453/v1 -> Enhancing Lithium Exploration in the Borborema Province, Northeast Brazil: Integrating Airborne Geophysics, Low-Density Geochemistry, and Machine Learning Algorithms
- https://www.researchgate.net/publication/362263694_Machine_Learning_Methods_for_Quantifying_Uncertainty_in_Prospectivity_Mapping_of_Magmatic-Hydrothermal_Gold_Deposits_A_Case_Study_from_Juruena_Mineral_Province_Northern_Mato_Grosso_Brazil
- https://www.researchgate.net/publication/360055592_Predicting_mineralization_and_targeting_exploration_criteria_based_on_machine-learning_in_the_Serra_de_Jacobina_quartz-pebble-metaconglomerate_Au-U_deposits_Sao_Francisco_Craton_Brazil
##### Fuzzy
- https://www.researchgate.net/publication/272170968_A_Comparative_Analysis_of_Weights_of_Evidence_Evidential_Belief_Functions_and_Fuzzy_Logic_for_Mineral_Potential_Mapping_Using_Incomplete_Data_at_the_Scale_of_Investigation A Comparative Analysis of Weights of Evidence, Evidential Belief Functions, and Fuzzy Logic for Mineral Potential Mapping Using Incomplete Data at the Scale of Investigation
- https://www.researchgate.net/publication/360386350_Application_of_Fuzzy_Gamma_Operator_to_Generate_Mineral_Prospectivity_Mapping_for_Cu-Mo_Porphyry_Deposits_Case_Study_Kighal-Bourmolk_Area_Northwestern_Iran
- https://www.researchgate.net/publication/348823482_Combining_fuzzy_analytic_hierarchy_process_with_concentration-area_fractal_for_mineral_prospectivity_mapping_A_case_study_involving_Qinling_orogenic_belt_in_central_China
- https://tupa.gtk.fi/raportti/arkisto/m60_2003_1.pdf -> Conceptual Fuzzy Logic Prospectivity Analysis of the Kuusamo Area
- https://www.researchgate.net/publication/356508827_Geophysical-spatial_Data_Modeling_using_Fuzzy_Logic_Applied_to_Nova_Aurora_Iron_District_Northern_Minas_Gerais_State_Southeastern_Brazil
- https://www.researchgate.net/publication/356937528_Mineral_prospectivity_mapping_a_potential_technique_for_sustainable_mineral_exploration_and_mining_activities_-_a_case_study_using_the_copper_deposits_of_the_Tagmout_basin_Morocco
#### Canada
- http://www.geosciencebc.com/i/pdf/SummaryofActivities2015/SoA2015_Granek.pdf -> Advanced Geoscience Targeting via Focused Machine Learning Applied to the QUEST Project Dataset, British Columbia
- https://open.library.ubc.ca/soa/cIRcle/collections/ubctheses/24/items/1.0340340 -> Application of machine learning algorithms to mineral prospectivity mapping
- https://www.researchgate.net/publication/369599705_A_study_of_faults_in_the_Superior_province_of_Ontario_and_Quebec_using_the_random_forest_machine_learning_algorithm_spatial_relationship_to_gold_mines
- A balanced mineral prospectivity model of Canadian magmatic Ni (± Cu ± Co ± PGE) sulphide mineral systems using conditional variational autoencoders -> https://www.sciencedirect.com/science/article/pii/S0169136824004621
- https://link.springer.com/article/10.1007/s11053-025-10468-z -> Class Label Representativeness in Machine Learning-Based Mineral Prospectivity Mapping
- https://www.researchgate.net/publication/273176257_Data-_and_Knowledge_driven_mineral_prospectivity_maps_for_Canada's_North
- https://www.researchgate.net/publication/300153215_Data_mining_for_real_mining_A_robust_algorithm_for_prospectivity_mapping_with_uncertainties
- https://www.sciencedirect.com/science/article/pii/S1674987123002268 -> Development and application of feature engineered geological layers for ranking magmatic, volcanogenic, and orogenic system components in Archean greenstone belts
- https://eartharxiv.org/repository/view/8898/ -> Geoscientific Input Feature Selection for CNN-driven Mineral Prospectivity Mapping
- https://www.researchgate.net/publication/343511849_Identification_of_intrusive_lithologies_in_volcanic_terrains_in_British_Columbia_by_machine_learning_using_Random_Forests_the_value_of_using_a_soft_classifier
- https://www.researchgate.net/publication/365782501_Improving_Mineral_Prospectivity_Model_Generalization_An_Example_from_Orogenic_Gold_Mineralization_of_the_Sturgeon_Lake_Transect_Ontario_Canada
- https://qspace.library.queensu.ca/bitstream/handle/1974/28138/Cevik_Ilkay_S_202009_MASc.pdf?sequence=3&isAllowed=y -> MACHINE LEARNING ENHANCEMENTS FOR KNOWLEDGE DISCOVERY IN MINERAL EXPLORATION AND IMPROVED MINERAL RESOURCE CLASSIFICATION
- https://www.researchgate.net/publication/348983384_Mineral_prospectivity_mapping_using_a_VNet_convolutional_neural_network
 - [corporate link](https://assets.website-files.com/60f4b75ca5b322cf572c1fde/612d5c5d73694b869a536bfa_VNET_TLE_paper.pdf)
 - https://www.researchgate.net/publication/369048379_Mineral_Prospectivity_Mapping_Using_Machine_Learning_Techniques_for_Gold_Exploration_in_the_Larder_Lake_Area_Ontario_Canada
 - https://link.springer.com/article/10.1007/s11053-024-10451-0 -> Mineral Prospectivity Modeling of Graphite Deposits and Occurrences in Canada
- https://www.researchgate.net/publication/337167506_Orogenic_gold_prospectivity_mapping_using_machine_learning
- https://www.researchgate.net/publication/290509352_Precursors_predicted_by_artificial_neural_networks_for_mass_balance_calculations_Quantifying_hydrothermal_alteration_in_volcanic_rocks 
- https://link.springer.com/article/10.1007/s11053-024-10438-x#Sec9 -> Pan-Canadian Predictive Modeling of Lithium–Cesium–Tantalum Pegmatites with Deep Learning and Natural Language Processing
- https://link.springer.com/article/10.1007/s11053-024-10369-7 -> Predictive Modeling of Canadian Carbonatite-Hosted REE +/− Nb Deposits
- https://www.sciencedirect.com/science/article/pii/S0098300422001406 -> Preliminary geological mapping with convolution neural network using statistical data augmentation on a 3D model
- https://www.researchgate.net/publication/352046255_Study_of_the_Influence_of_Non-Deposit_Locations_in_Data-Driven_Mineral_Prospectivity_Mapping_A_Case_Study_on_the_Iskut_Project_in_Northwestern_British_Columbia_Canada
- https://www.researchgate.net/publication/220164155_Support_vector_machine_A_tool_for_mapping_mineral_prospectivity
- https://www.researchgate.net/publication/348111963_Support_Vector_Machine_and_Artificial_Neural_Network_Modelling_of_Orogenic_Gold_Prospectivity_Mapping_in_the_Swayze_greenstone_belt_Ontario_Canada
 - PhD thesis -> https://zone.biblio.laurentian.ca/bitstream/10219/3736/1/PhD%20Thesis%20Maepa_20210603.%281%29.pdf -> Exploration targeting for gold deposits using spatial data analytics, machine learning and deep transfer learning in the Swayze and Matheson greenstone belts, Ontario, Canada
- https://data.geology.gov.yk.ca/Reference/95936#InfoTab -> Updates to the Yukon Geological Survey’s mineral potential mapping methodology
- http://www.geosciencebc.com/i/pdf/SummaryofActivities2015/SoA2015_Granek.pdf -> Advanced Geoscience Targeting via Focused Machine Learning Applied to the QUEST Project Dataset, British Columbia
#### Central Africa
- https://www.researchgate.net/publication/323452014_The_Utility_of_Machine_Learning_in_Identification_of_Key_Geophysical_and_Geochemical_Datasets_A_Case_Study_in_Lithological_Mapping_in_the_Central_African_Copper_Belt
- https://www.researchgate.net/publication/334436808_Lithological_Mapping_in_the_Central_African_Copper_Belt_using_Random_Forests_and_Clustering_Strategies_for_Optimised_Results
#### Chile
- https://www.researchgate.net/publication/341485750_Evaluation_of_random_forest-based_analysis_for_the_gypsum_distribution_in_the_Atacama_desert
#### China
- https://www.researchgate.net/publication/374968979_3D_cooperative_inversion_of_airborne_magnetic_and_gravity_gradient_data_using_deep_learning_techniques - 3D cooperative inversion of airborne magnetic and gravity gradient data using deep learning techniques [UNSEEN]
- https://www.researchgate.net/publication/369919958_3D_mineral_exploration_Cu-Zn_targeting_with_multi-source_geoscience_datasets_in_the_Weilasituo-bairendaba_district_Inner_Mongolia_China
- https://www.researchgate.net/publication/350817136_3D_Mineral_Prospectivity_Mapping_Based_on_Deep_Metallogenic_Prediction_Theory_A_Case_Study_of_the_Lala_Copper_Mine_Sichuan_China
- https://www.researchgate.net/publication/336771580_3D_Mineral_Prospectivity_Mapping_with_Random_Forests_A_Case_Study_of_Tongling_Anhui_China
- https://www.sciencedirect.com/science/article/pii/S0169136823005772 -> 3D mineral prospectivity modeling in the Sanshandao goldfield, China using the convolutional neural network with attention mechanism
- https://www.sciencedirect.com/science/article/pii/S0009281924001144 -> 3D mineral prospectivity modeling using deep adaptation network transfer learning: A case study of the Xiadian gold deposit, Eastern China
- https://www.sciencedirect.com/science/article/pii/S0009281924000497 -> 3D mineral prospectivity modeling using multi-scale 3D convolution neural network and spatial attention approaches
- https://www.researchgate.net/publication/366201930_3D_Quantitative_Metallogenic_Prediction_of_Indium-Rich_Ore_Bodies_in_the_Dulong_Sn-Zn_Polymetallic_Deposit_Yunnan_Province_SW_China
- https://www.researchgate.net/publication/329600793_A_combined_approach_using_spatially-weighted_principal_components_analysis_and_wavelet_transformation_for_geochemical_anomaly_mapping_in_the_Dashui_ore-concentration_district_Central_China
- https://www.researchgate.net/publication/349034539_A_Comparative_Study_of_Machine_Learning_Models_with_Hyperparameter_Optimization_Algorithm_for_Mapping_Mineral_Prospectivity
- https://www.researchgate.net/publication/354132594_A_Convolutional_Neural_Network_of_GoogLeNet_Applied_in_Mineral_Prospectivity_Prediction_Based_on_Multi-source_Geoinformation
- https://www.researchgate.net/publication/369865076_A_deep-learning-based_mineral_prospectivity_modeling_framework_and_workflow_in_prediction_of_porphyry-epithermal_mineralization_in_the_Duolong_Ore_District_Tibet
- https://www.researchgate.net/publication/374982967_A_Framework_for_Data-Driven_Mineral_Prospectivity_Mapping_with_Interpretable_Machine_Learning_and_Modulated_Predictive_Modeling
- https://www.sciencedirect.com/science/article/pii/S0169136824002026 -> A Global-Local collaborative approach to quantifying spatial non-stationarity in three-dimensional mineral prospectivity modeling
- https://link.springer.com/article/10.1007/s11053-024-10344-2 -> A Heterogeneous Graph Construction Method for Mineral Prospectivity Mapping [UNSEEN]
- https://www.researchgate.net/publication/353421842_A_hybrid_logistic_regression_gene_expression_programming_model_and_its_application_to_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/375764940_A_lightweight_convolutional_neural_network_with_end-to-end_learning_for_three-dimensional_mineral_prospectivity_modeling_A_case_study_of_the_Sanhetun_Area_Heilongjiang_Province_Northeastern_China
- https://www.researchgate.net/publication/339821823_A_Monte_Carlo-based_framework_for_risk-return_analysis_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/373715610_A_Multimodal_Learning_Framework_for_Comprehensive_3D_Mineral_Prospectivity_Modeling_with_Jointly_Learned_Structure-Fluid_Relationships
- https://www.sciencedirect.com/science/article/pii/S0169136824001343 -> A novel hybrid ensemble model for mineral prospectivity prediction: A case study in the Malipo W-Sn mineral district, Yunnan Province, China
- https://www.researchgate.net/publication/347344551_A_positive_and_unlabeled_learning_algorithm_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/335036019_An_Autoencoder-Based_Dimensionality_Reduction_Algorithm_for_Intelligent_Clustering_of_Mineral_Deposit_Data
- https://www.researchgate.net/publication/363696083_An_Integrated_Framework_for_Data-Driven_Mineral_Prospectivity_Mapping_Using_Bagging-Based_Positive_Unlabeled_Learning_and_Bayesian_Cost-Sensitive_Logistic_Regression
- https://link.springer.com/article/10.1007/s11053-024-10349-x -> An Uncertainty-Quantification Machine Learning Framework for Data-Driven Three-Dimensional Mineral Prospectivity Mapping
- https://link.springer.com/article/10.1007/s11004-023-10076-8 - An Interpretable Graph Attention Network for Mineral Prospectivity Mapping
- https://www.researchgate.net/publication/332751556_Application_of_hierarchical_clustering_singularity_mapping_and_Kohonen_neural_network_to_identify_Ag-Au-Pb-Zn_polymetallic_mineralization_associated_geochemical_anomaly_in_Pangxidong_district
- https://www.mdpi.com/2075-163X/14/9/945 -> Application of Machine Learning to Characterize Metallogenic Potential Based on Trace Elements of Zircon: A Case Study of the Tethyan Domain
- https://www.researchgate.net/publication/339096362_Application_of_nonconventional_mineral_exploration_techniques_case_studies
- https://www.researchgate.net/publication/325702993_Assessment_of_Geochemical_Anomaly_Uncertainty_Through_Geostatistical_Simulation_and_Singularity_Analysis
- https://www.researchgate.net/publication/368586826_Bagging-based_Positive-Unlabeled_Data_Learning_Algorithm_with_Base_Learners_Random_Forest_and_XGBoost_for_3D_Exploration_Targeting_in_the_Kalatongke_District_Xinjiang_China
- https://link.springer.com/article/10.1007/s11004-024-10153-6 -> Causal Discovery and Deep Learning Algorithms for Detecting Geochemical Patterns Associated with Gold-Polymetallic Mineralization: A Case Study of the Edongnan Region [UNSEEN]
- https://www.sciencedirect.com/science/article/pii/S0169136824001409 -> CNN-Transformers for mineral prospectivity mapping in the Maodeng–Baiyinchagan area, Southern Great Xing'an Range
- https://www.researchgate.net/publication/347079505_Convolutional_neural_network_and_transfer_learning_based_mineral_prospectivity_modeling_for_geochemical_exploration_of_Au_mineralization_within_the_Guandian-Zhangbaling_area_Anhui_Province_China
- https://www.researchgate.net/publication/352703015_Data-driven_based_logistic_function_and_prediction-area_plot_for_mineral_prospectivity_mapping_a_case_study_from_the_eastern_margin_of_Qinling_orogenic_belt_central_China
- https://www.sciencedirect.com/science/article/abs/pii/S0012825218306123 -> Deep learning and its application in geochemical mapping
- https://www.frontiersin.org/articles/10.3389/feart.2024.1308426/full -> Deep gold prospectivity modeling in the Jiaojia gold belt, Jiaodong Peninsula, eastern China using machine learning of geometric and geodynamic variables
- https://www.researchgate.net/publication/352893038_Detection_of_geochemical_anomalies_related_to_mineralization_using_the_GANomaly_network
- https://www.researchgate.net/publication/357685352_Determination_of_Predictive_Variables_in_Mineral_Prospectivity_Mapping_Using_Supervised_and_Unsupervised_Methods
- https://www.sciencedirect.com/science/article/abs/pii/S0375674221001370 -> Distinguishing IOCG and IOA deposits via random forest algorithm based on magnetite composition
- https://www.researchgate.net/publication/340401748_Effects_of_Random_Negative_Training_Samples_on_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/360333702_Ensemble_learning_models_with_a_Bayesian_optimization_algorithm_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/267927676_Evaluation_of_uncertainty_in_mineral_prospectivity_mapping_due_to_missing_evidence_A_case_study_with_skarn-type_Fe_deposits_in_Southwestern_Fujian_Province_China
- https://www.mdpi.com/2075-163X/14/5/492 ->Exploration Vectors and Indicators Extracted by Factor Analysis and Association Rule Algorithms at the Lintan Carlin-Type Gold Deposit, Youjiang Basin, China
- https://www.researchgate.net/publication/379852209_Fractal-Based_Multi-Criteria_Feature_Selection_to_Enhance_Predictive_Capability_of_AI-Driven_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/338789096_From_2D_to_3D_Modeling_of_Mineral_Prospectivity_Using_Multi-source_Geoscience_Datasets_Wulong_Gold_District_China
- https://www.researchgate.net/publication/359714254_Geochemical_characterization_of_the_Central_Mineral_Belt_U_Cu_Mo_V_mineralization_Labrador_Canada_Application_of_unsupervised_machine-learning_for_evaluation_of_IOCG_and_affiliated_mineral_potential
- https://www.researchgate.net/publication/350788828_Geochemically_Constrained_Prospectivity_Mapping_Aided_by_Unsupervised_Cluster_Analysis
- https://www.researchgate.net/publication/267927506_GIS-based_mineral_potential_modeling_by_advanced_spatial_analytical_methods_in_the_southeastern_Yunnan_mineral_district_China
- https://www.researchgate.net/publication/380190183_Geologically_Constrained_Convolutional_Neural_Network_for_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/332997161_GNER_A_Generative_Model_for_Geological_Named_Entity_Recognition_Without_Labeled_Data_Using_Deep_Learning
- https://www.researchgate.net/publication/307011381_Identification_and_mapping_of_geochemical_patterns_and_their_significance_for_regional_metallogeny_in_the_southern_Sanjiang_China
- https://link.springer.com/article/10.1007/s11053-024-10334-4 -> Identification of Geochemical Anomalies Using an End-to-End Transformer
- https://www.researchgate.net/publication/359627130_Identification_of_ore-finding_targets_using_the_anomaly_components_of_ore-forming_element_associations_extracted_by_SVD_and_PCA_in_the_Jiaodong_gold_cluster_area_Eastern_China
- https://www.researchgate.net/publication/282621670_Identifying_geochemical_anomalies_associated_with_Au-Cu_mineralization_using_multifractal_and_artificial_neural_network_models_in_the_Ningqiang_district_Shaanxi_China
- https://www.sciencedirect.com/science/article/abs/pii/S0375674224000943 -> Integrate physics-driven dynamics simulation with data-driven machine learning to predict potential targets in maturely explored orefields: A case study in Tongguangshan orefield, Tongling, China
- https://www.researchgate.net/publication/329299202_Integrating_sequential_indicator_simulation_and_singularity_analysis_to_analyze_uncertainty_of_geochemical_anomaly_for_exploration_targeting_of_tungsten_polymetallic_mineralization_Nanling_belt_South_
- https://www.sciencedirect.com/science/article/abs/pii/S0883292724001987 -> Integrating soil geochemistry and machine learning for enhanced mineral exploration at the dayu gold deposit, south China block
- https://www.mdpi.com/2071-1050/15/13/10269 -> Intelligent Identification and Prediction Mineral Resources Deposit Based on Deep Learning
- https://link.springer.com/article/10.1007/s11053-024-10396-4 -> Interpretable SHAP Model Combining Meta-learning and Vision Transformer for Lithology Classification Using Limited and Unbalanced Drilling Data in Well Logging
- https://www.researchgate.net/publication/358555996_Learning_3D_mineral_prospectivity_from_3D_geological_models_using_convolutional_neural_networks_Application_to_a_structure-controlled_hydrothermal_gold_deposit
- https://www.researchgate.net/publication/352476625_Machine_Learning-Based_3D_Modeling_of_Mineral_Prospectivity_Mapping_in_the_Anqing_Orefield_Eastern_China
- https://www.mdpi.com/2076-3417/15/8/4082 -> Machine-Learning-Based Integrated Mining Big Data and Multi-Dimensional Ore-Forming Prediction: A Case Study of Yanshan Iron Mine, Hebei, China
- https://www.researchgate.net/publication/331575655_Mapping_Geochemical_Anomalies_Through_Integrating_Random_Forest_and_Metric_Learning_Methods
- https://www.researchgate.net/publication/229399579_Mapping_geochemical_singularity_using_multifractal_analysis_Application_to_anomaly_definition_on_stream_sediments_data_from_Funin_Sheet_Yunnan_China
- https://www.researchgate.net/publication/328255422_Mapping_mineral_prospectivity_through_big_data_analytics_and_a_deep_learning_algorithm
- https://www.researchgate.net/publication/334106787_Mapping_Mineral_Prospectivity_via_Semi-supervised_Random_Forest
- https://www.researchgate.net/publication/236270466_Mapping_of_district-scale_potential_targets_using_fractal_models
- https://www.researchgate.net/publication/357584076_Mapping_prospectivity_for_regolith-hosted_REE_deposits_via_convolutional_neural_network_with_generative_adversarial_network_augmented_data
- https://www.researchgate.net/publication/328623280_Maximum_Entropy_and_Random_Forest_Modeling_of_Mineral_Potential_Analysis_of_Gold_Prospectivity_in_the_Hezuo-Meiwu_District_West_Qinling_Orogen_China
- https://www.sciencedirect.com/science/article/pii/S016913682400163X -> Metallogenic prediction based on fractal theory and machine learning in Duobaoshan Area, Heilongjiang Province
- https://www.sciencedirect.com/science/article/pii/S0169136824003810 -> Mineral prospectivity mapping susceptibility evaluation based on interpretable ensemble learning
- https://link.springer.com/article/10.1007/s11053-024-10386-6 -> Mineral Prospectivity Mapping Based on Spatial Feature Classification with Geological Map Knowledge Graph Embedding: Case Study of Gold Ore Prediction at Wulonggou, Qinghai Provinc
- https://www.researchgate.net/publication/235443301_Mineral_potential_mapping_in_a_frontier_region
- https://www.researchgate.net/publication/235443302_Mineral_potential_mapping_in_frontier_regions_A_Mongolian_case_study
- https://www.researchgate.net/publication/369104190_Mineral_Prospectivity_Mapping_Using_Attention-based_Convolutional_Neural_Network 
- https://www.nature.com/articles/s41598-024-73357-0 -> Mineral prospectivity prediction based on convolutional neural network and ensemble learning
- https://www.researchgate.net/publication/329037175_Mineral_prospectivity_analysis_for_BIF_iron_deposits_A_case_study_in_the_Anshan-Benxi_area_Liaoning_province_North-East_China
- https://link.springer.com/article/10.1007/s11053-024-10335-3 -> Mineral Prospectivity Prediction Based on Self-Supervised Contrastive Learning and Geochemical Data: A Case Study of the Gold Deposit in the Malanyu District, Hebei Province, China [USEEN]
- https://www.researchgate.net/publication/377694139_Manganese_mineral_prospectivity_based_on_deep_convolutional_neural_networks_in_Songtao_of_northeastern_Guizhou
- https://www.researchgate.net/publication/ 351649498_Mineral_Prospectivity_Mapping_based_on_Isolation_Forest_and_Random_Forest_Implication_for_the_Existence_of_Spatial_Signature_of_Mineralization_in_Outliers
- https://www.researchgate.net/publication/358528670_Mineral_Prospectivity_Mapping_Based_on_Wavelet_Neural_Network_and_Monte_Carlo_Simulations_in_the_Nanling_W-Sn_Metallogenic_Province
- https://www.researchgate.net/publication/352983697_Mineral_prospectivity_mapping_by_deep_learning_method_in_Yawan-Daqiao_area_Gansu
- https://www.researchgate.net/publication/367106018_Mineral_Prospectivity_Mapping_of_Porphyry_Copper_Deposits_Based_on_Remote_Sensing_Imagery_and_Geochemical_Data_in_the_Duolong_Ore_District_Tibet - Mineral Prospectivity Mapping of Porphyry Copper Deposits Based on Remote Sensing Imagery and Geochemical Data in the Duolong Ore District, Tibet
- https://www.researchgate.net/publication/355749736_Mineral_prospectivity_mapping_using_a_joint_singularity-based_weighting_method_and_long_short-term_memory_network
- https://www.researchgate.net/publication/369104190_Mineral_Prospectivity_Mapping_Using_Attention-based_Convolutional_Neural_Network 
- https://www.researchgate.net/publication/365434839_Mineral_Prospectivity_Mapping_Using_Deep_Self-Attention_Model
- https://www.researchgate.net/publication/379674196_Mineral_prospectivity_mapping_using_knowledge_embedding_and_explainable_ensemble_learning_A_case_study_of_the_Keeryin_ore_concentration_in_Sichuan_China
- Mineral Prospectivity Mapping Using Semi-supervised Machine Learning -> https://link.springer.com/article/10.1007/s11004-024-10161-6
- https://www.researchgate.net/publication/350817877_Mineral_Prospectivity_Prediction_via_Convolutional_Neural_Networks_Based_on_Geological_Big_Data
- https://www.researchgate.net/publication/338871759_Modeling-based_mineral_system_approach_to_prospectivity_mapping_of_stratabound_hydrothermal_deposits_A_case_study_of_MVT_Pb-Zn_deposits_in_the_Huayuan_area_northwestern_Hunan_Province_China
- https://www.sciencedirect.com/science/article/pii/S0169136824003172 -> New insights into the metallogenic genesis of the Xiadian Au deposit, Jiaodong Peninsula, Eastern China: Constraints from integrated rutile in-situ geochemical analysis and machine learning discrimination
- https://www.researchgate.net/publication/332547136_Prospectivity_Mapping_for_Porphyry_Cu-Mo_Mineralization_in_the_Eastern_Tianshan_Xinjiang_Northwestern_China
- https://www.sciencedirect.com/science/article/pii/S2666544125000607 -> Quantifying uncertainty of mineral prediction using a novel Bayesian deep learning framework
- https://www.sciencedirect.com/science/article/pii/S0169136824001823 -> Quantitative prediction methods and applications of digital ore deposit models
- https://www.researchgate.net/publication/344303914_Random-Drop_Data_Augmentation_of_Deep_Convolutional_Neural_Network_for_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/371044606_Supervised_Mineral_Prospectivity_Mapping_via_Class-Balanced_Focal_Loss_Function_on_Imbalanced_Geoscience_DatasetsSupervised Mineral Prospectivity Mapping via Class-Balanced Focal Loss Function on Imbalanced Geoscience Datasets
- https://www.researchgate.net/publication/361520562_Recognizing_Multivariate_Geochemical_Anomalies_Related_to_Mineralization_by_Using_Deep_Unsupervised_Graph_Learning
- https://www.sciencedirect.com/science/article/pii/S0169136824003937 -> Semi-supervised graph convolutional networks for integrating continuous and binary evidential layers for mineral exploration targeting
- https://www.sciencedirect.com/science/article/pii/S0009281924001375 -> Spatial weighting — An effective incorporation of geological expertise into deep learning models
- https://www.researchgate.net/publication/371044606_Supervised_Mineral_Prospectivity_Mapping_via_Class-Balanced_Focal_Loss_Function_on_Imbalanced_Geoscience_Datasets
- https://www.researchgate.net/publication/360028637_Three-Dimensional_Mineral_Prospectivity_Mapping_by_XGBoost_Modeling_A_Case_Study_of_the_Lannigou_Gold_Deposit_China
- https://link.springer.com/article/10.1007/s11053-024-10387-5 - Toward Data-Driven Mineral Prospectivity Mapping from Remote Sensing Data Using Deep Forest Predictive Model
- https://www.researchgate.net/publication/361589587_Unlabeled_Sample_Selection_for_Mineral_Prospectivity_Mapping_by_Semi-supervised_Support_Vector_Machine
- https://www.researchgate.net/publication/343515866_Using_deep_variational_autoencoder_networks_for_recognizing_geochemical_anomalies
- https://link.springer.com/article/10.1007/s11004-024-10151-8 -> Using Three-dimensional Modeling and Random Forests to Predict Deep Ore Potentials: A Case Study on Xiongcun Porphyry Copper–Gold Deposit in Tibet, China
- https://www.researchgate.net/publication/361194407_Visual_Interpretable_Deep_Learning_Algorithm_for_Geochemical_Anomaly_Recognition
#### Egypt
- https://www.researchgate.net/publication/340084035_Reliability_of_using_ASTER_data_in_lithologic_mapping_and_alteration_mineral_detection_of_the_basement_complex_of_West_Berenice_Southeastern_Desert_Egypt
#### England
- https://www.researchgate.net/publication/342339753_A_machine_learning_approach_to_tungsten_prospectivity_modelling_using_knowledge-driven_feature_extraction_and_model_confidence
- https://www.researchgate.net/project/Enhancing-the-Geological-Understanding-of-SW-England-Using-Machine-Learning-Algorithms
#### Eritrea
- https://www.researchgate.net/publication/349158008_Mapping_gold_mineral_prospectivity_based_on_weights_of_evidence_method_in_southeast_Asmara_Eritrea
#### Finland
- https://www.researchgate.net/publication/360661926_Target-scale_prospectivity_modeling_for_gold_mineralization_within_the_Rajapalot_Au-Co_project_area_in_northern_Fennoscandian_Shield_Finland_Part_2_Application_of_self-organizing_maps_and_artificial_n
- https://www.sciencedirect.com/science/article/pii/S0169136824004037 -> Addressing imbalanced data for machine learning based mineral prospectivity mapping
- https://publications.csiro.au/publications/#publication/PIcsiro:EP146125/SQmineral%20prospectivity/RP1/RS50/RORECENT/STsearch-by-keyword/LISEA/RI12/RT26 -> A novel spatial analysis approach for assessing regional-scale mineral prospectivity In Northern Finland
- https://www.researchgate.net/publication/332352805_Boosting_for_Mineral_Prospectivity_Modeling_A_New_GIS_Toolbox
- https://www.researchgate.net/publication/324517415_Can_boosting_boost_minimal_invasive_exploration_targeting
- https://www.researchgate.net/publication/248955109_Combined_conceptualempirical_prospectivity_mapping_for_orogenic_gold_in_the_northern_Fennoscandian_Shield_Finland
- https://www.researchgate.net/publication/283451958_Data-driven_logistic-based_weighting_of_geochemical_and_geological_evidence_layers_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/320280611_Evaluation_of_boosting_algorithms_for_prospectivity_mapping
- https://www.researchgate.net/publication/298297988_Fuzzy_logic_data_integration_technique_used_as_a_nickel_exploration_tool
- https://www.researchgate.net/publication/259372191_Gravity_data_in_regional_scale_3D_and_gold_prospectivity_modelling_-_example_from_the_Central_Lapland_greenstone_belt_northern_Finland
- https://www.researchgate.net/publication/315381587_Introduction_to_the_special_issue_GIS-based_mineral_potential_targeting
- https://www.researchgate.net/publication/320709733_Knowledge-driven_prospectivity_model_for_Iron_oxide-Cu-Au_IOCG_deposits_in_northern_Finland
- https://tupa.gtk.fi/raportti/arkisto/57_2021.pdf -> Mineral Prospectivity and Exploration Targeting MinProXT 2021 Webinar - paper compilation
- https://tupa.gtk.fi/raportti/arkisto/29_2023.pdf -> Mineral Prospectivity and Exploration Targeting MinProXT 2022 Webinar - paper compilation
- https://www.researchgate.net/publication/312180531_Optimizing_a_Knowledge-driven_Prospectivity_Model_for_Gold_Deposits_Within_Perapohja_Belt_Northern_Finland
- https://www.researchgate.net/publication/320703774_Prospectivity_Models_for_Volcanogenic_Massive_Sulfide_Deposits_VMS_in_Northern_Finland
- https://www.researchgate.net/publication/280875727_Receiver_operating_characteristics_ROC_as_validation_tool_for_prospectivity_models_-_A_magmatic_Ni-Cu_case_study_from_the_Central_Lapland_Greenstone_Belt_Northern_Finland
- https://www.researchgate.net/publication/332298116_Scalability_of_the_Mineral_Prospectivity_Modelling_-_An_orogenic_gold_case_study_from_northern_Finland
- https://www.researchgate.net/publication/251786465_Spatial_data_analysis_as_a_tool_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/331006924_Unsupervised_clustering_and_empirical_fuzzy_memberships_for_mineral_prospectivity_modelling
#### Ghana
- https://www.researchgate.net/publication/227256267_Application_of_Data-Driven_Evidential_Belief_Functions_to_Prospectivity_Mapping_for_Aquamarine-Bearing_Pegmatites_Lundazi_District_Zambia
- https://www.researchgate.net/publication/226842511_Mapping_of_prospectivity_and_estimation_of_number_of_undiscovered_prospects_for_lode_gold_southwestern_Ashanti_Belt_Ghana
- https://www.researchgate.net/publication/233791624_Spatial_association_of_gold_deposits_with_remotely_-_sensed_faults_South_Ashanti_belt_Ghana
#### Greenland
- https://www.researchgate.net/publication/360970965_Identification_of_Radioactive_Mineralized_Lithology_and_Mineral_Prospectivity_Mapping_Based_on_Remote_Sensing_in_High-Latitude_Regions_A_Case_Study_on_the_Narsaq_Region_of_Greenland
#### India
- https://www.researchgate.net/publication/226092981_A_Hybrid_Neuro-Fuzzy_Model_for_Mineral_Potential_Mapping
- https://www.researchgate.net/publication/225328359_A_Hybrid_Fuzzy_Weights-of-Evidence_Model_for_Mineral_Potential_Mapping
- https://www.researchgate.net/publication/227221497_Artificial_Neural_Networks_for_Mineral-Potential_Mapping_A_Case_Study_from_Aravalli_Province_Western_India
- https://www.researchgate.net/publication/222050039_Bayesian_network_classifiers_for_mineral_potential_mapping
- https://www.researchgate.net/publication/355397149_Gold_Prospectivity_Mapping_in_the_Sonakhan_Greenstone_Belt_Central_India_A_Knowledge-Driven_Guide_for_Target_Delineation_in_a_Region_of_Low_Exploration_Maturity
- https://www.researchgate.net/publication/272092276_Extended_Weights-of-Evidence_Modelling_for_Predictive_Mapping_of_Base_Metal_Deposit_Potential_in_Aravalli_Province_Western_India
- https://www.researchgate.net/publication/226193283_Knowledge-Driven_and_Data-Driven_Fuzzy_Models_for_Predictive_Mineral_Potential_Mapping
- https://www.researchgate.net/publication/238027981_SVM-based_base-metal_prospectivity_modeling_of_the_Aravalli_Orogen_Northwestern_India
- https://www.researchgate.net/publication/372636338_Unsupervised_machine_learning_based_prospectivity_analysis_of_NW_and_NE_India_for_carbonatite-alkaline_complex-related_REE_deposits
#### Indonesia
- https://www.researchgate.net/publication/263542819_Regional-Scale_Geothermal_Prospectivity_Mapping_in_West_Java_Indonesia_by_Data-driven_Evidential_Belief_Functions
#### Iran
- https://www.researchgate.net/publication/390072341_Enhancing_regional-scale_Pb-Zn_prospectivity_mapping_through_data_augmentation_Joint_application_of_unsupervised_random_forests_and_convolutional_neural_network
- https://www.researchgate.net/publication/325697373_A_comparative_analysis_of_artificial_neural_network_ANN_wavelet_neural_network_WNN_and_support_vector_machine_SVM_data-driven_models_to_mineral_potential_mapping_for_copper_mineralizations_in_the_Shah 
- https://www.researchgate.net/publication/358507255_A_Comparative_Study_of_Convolutional_Neural_Networks_and_Conventional_Machine_Learning_Models_for_Lithological_Mapping_Using_Remote_Sensing_Data
- https://www.researchgate.net/publication/351750324_A_data_augmentation_approach_to_XGboost-based_mineral_potential_mapping_An_example_of_carbonate-hosted_Zn_Pb_mineral_systems_of_Western_Iran
- https://www.researchgate.net/publication/336471932_A_knowledge-guided_fuzzy_inference_approach_for_integrating_geophysics_geochemistry_and_geology_data_in_deposit-scale_porphyry_copper_targeting_Saveh-Iran
- https://www.researchgate.net/publication/348500913_A_new_strategy_for_spatial_predictive_mapping_of_mineral_prospectivity
- https://www.researchgate.net/publication/348482539_A_new_strategy_for_spatial_predictive_mapping_of_mineral_prospectivity_Automated_hyperparameter_tuning_of_random_forest_approach
- https://www.researchgate.net/publication/352251016_A_simulation-based_framework_for_modulating_the_effects_of_subjectivity_in_greenfield_Mineral_Prospectivity_Mapping_with_geochemical_and_geological_data
- https://www.researchgate.net/publication/296638839_An_AHP-TOPSIS_Predictive_Model_for_District-Scale_Mapping_of_Porphyry_Cu-Au_Potential_A_Case_Study_from_Salafchegan_Area_Central_Iran
- https://www.researchgate.net/publication/278029106_Application_of_Discriminant_Analysis_and_Support_Vector_Machine_in_Mapping_Gold_Potential_Areas_for_Further_Drilling_in_the_Sari-Gunay_Gold_Deposit_NW_Iran
- https://www.researchgate.net/publication/220164381_Application_of_geochemical_zonality_coefficients_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/330359897_Application_of_hybrid_AHP-TOPSIS_method_for_prospectivity_modeling_of_Cu_porphyry_in_Varzaghan_district_Iran
- https://www.researchgate.net/publication/356872819_Application_of_self-organizing_map_SOM_and_K-means_clustering_algorithms_for_portraying_geochemical_anomaly_patterns_in_Moalleman_district_NE_Iran
- https://www.researchgate.net/publication/258505300_Application_of_staged_factor_analysis_and_logistic_function_to_create_a_fuzzy_stream_sediment_geochemical_evidence_layer_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/358567148_Applications_of_data_augmentation_in_mineral_prospectivity_prediction_based_on_convolutional_neural_networks
- https://www.researchgate.net/publication/353761696_Assessing_the_effects_of_mineral_systems-derived_exploration_targeting_criteria_for_Random_Forests-based_predictive_mapping_of_mineral_prospectivity_in_Ahar-Arasbaran_area_Iran
- https://www.sciencedirect.com/science/article/abs/pii/S0375674225000871 -> A semi-supervised learning framework for intelligent mineral prospectivity mapping: Incorporation of the CatBoost and Gaussian mixture model algorithms [UNSEEN]
- https://www.researchgate.net/publication/270586282_Data-Driven_Index_Overlay_and_Boolean_Logic_Mineral_Prospectivity_Modeling_in_Greenfields_Exploration
- https://www.researchgate.net/publication/356660905_Deep_GMDH_Neural_Networks_for_Predictive_Mapping_of_Mineral_Prospectivity_in_Terrains_Hosting_Few_but_Large_Mineral_Deposits
- https://link.springer.com/article/10.1007/s12145-025-01708-0 Density based spatial clustering of applications with noise and fuzzy C-means algorithms for unsupervised mineral prospectivity mapping [UNSEEN]
- https://www.researchgate.net/publication/317240761_Enhancement_and_Mapping_of_Weak_Multivariate_Stream_Sediment_Geochemical_Anomalies_in_Ahar_Area_NW_Iran
- https://www.sciencedirect.com/science/article/pii/S0009281924001223 -> Enhancing training performance of convolutional neural network algorithm through an autoencoder-based unsupervised labeling framework for mineral exploration targeting
- https://www.researchgate.net/publication/356580903_Evidential_data_integration_to_produce_porphyry_Cu_prospectivity_map_using_a_combination_of_knowledge_and_data_driven_methods
- https://research-repository.uwa.edu.au/en/publications/exploration-feature-selection-applied-to-hybrid-data-integration-Exploration feature selection applied to hybrid data integrationmodeling: Targeting copper-gold potential in central 
- https://www.researchgate.net/publication/333199619_Incorporation_of_principal_component_analysis_geostatistical_interpolation_approaches_and_frequency-space-based_models_for_portraying_the_Cu-Au_geochemical_prospects_in_the_Feizabad_district_NW_Iran
- https://www.researchgate.net/publication/351965039_Intelligent_geochemical_exploration_modeling_using_multiclass_support_vector_machine_and_integration_it_with_continuous_genetic_algorithm_in_Gonabad_region_Khorasan_Razavi_Iran
- https://www.researchgate.net/publication/310658663_Multifractal_interpolation_and_spectrum-area_fractal_modeling_of_stream_sediment_geochemical_data_Implications_for_mapping_exploration_targets
- https://www.researchgate.net/publication/267635150_Multivariate_regression_analysis_of_lithogeochemical_data_to_model_subsurface_mineralization_A_case_study_from_the_Sari_Gunay_epithermal_gold_deposit_NW_Iran
- https://www.researchgate.net/publication/330129457_Performance_evaluation_of_RBF-_and_SVM-based_machine_learning_algorithms_for_predictive_mineral_prospectivity_modeling_integration_of_S-A_multifractal_model_and_mineralization_controls
- https://www.researchgate.net/publication/353982380_Porphyry_Cu-Au_prospectivity_modelling_using_semi-supervised_learning_algorithm_in_Dehsalm_district_eastern_Iran_In_Farsi_with_extended_English_abstract
- https://www.researchgate.net/publication/320886789_Prospectivity_analysis_of_orogenic_gold_deposits_in_Saqez-Sardasht_Goldfield_Zagros_Orogen_Iran
- https://www.researchgate.net/publication/361529867_Prospectivity_mapping_of_orogenic_lode_gold_deposits_using_fuzzy_models_a_case_study_of_Saqqez_area_NW_of_Iran
- https://www.researchgate.net/publication/361717490_Quantifying_Uncertainties_Linked_to_the_Diversity_of_Mathematical_Frameworks_in_Knowledge-Driven_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/374730424_Recognition_of_mineralization-related_anomaly_patterns_through_an_autoencoder_neural_network_for_mineral_exploration_targeting
- https://www.researchgate.net/publication/349957803_Regional-Scale_Mineral_Prospectivity_Mapping_Support_Vector_Machines_and_an_Improved_Data-Driven_Multi-criteria_Decision-Making_Technique
- https://www.researchgate.net/publication/339153591_Sensitivity_analysis_of_prospectivity_modeling_to_evidence_maps_Enhancing_success_of_targeting_for_epithermal_gold_Takab_district_NW_Iran
- https://www.researchgate.net/publication/321076980_Spatial_analyses_of_exploration_evidence_data_to_model_skarn-type_copper_prospectivity_in_the_Varzaghan_district_NW_Iran
- https://www.researchgate.net/publication/304904242_Stepwise_regression_for_recognition_of_geochemical_anomalies_Case_study_in_Takab_area_NW_Iran
- https://www.researchgate.net/publication/350423220_Supervised_mineral_exploration_targeting_and_the_challenges_with_the_selection_of_deposit_and_non-deposit_sites_thereof
- https://www.sciencedirect.com/science/article/pii/S0009281924000801 -> Targeting porphyry Cu deposits in the Chahargonbad region of Iran: A joint application of deep belief networks and random forest techniques
- https://www.researchgate.net/publication/307874730_The_use_of_decision_tree_induction_and_artificial_neural_networks_for_recognizing_the_geochemical_distribution_patterns_of_LREE_in_the_Choghart_deposit_Central_Iran
- https://www.researchsquare.com/article/rs-4760956/v1 -> Uncertainty reduction with Hyperparameter Optimization in mineral prospectivity mapping: A Regularized Artificial Neural Network approach	[UNSEEN]
#### Ireland
- https://www.gsi.ie/en-ie/programmes-and-projects/tellus/activities/tellus-product-development/mineral-prospectivity/Pages/default.aspx - > NW Midlands Mineral Prospectivity Mapping
#### Nigeria
- https://www.researchgate.net/publication/390597293_Assessment_of_Mineral_Deposits_in_Part_of_North_Senatorial_Zone_Adamawa_State_Nigeria_Using_Remote_Sensing_Geographic_Information_Systems_and_Machine_Learning
#### Norway
- https://www.mdpi.com/2075-163X/9/2/131/htm - Prospectivity Mapping of Mineral Deposits in Northern Norway Using Radial Basis Function Neural Networks
#### Pakistan
- https://onlinelibrary.wiley.com/doi/full/10.1002/eng2.13031 _> Advanced Mineral Deposit Mapping via Deep Learning and SVM Integration With Remote Sensing Imaging Data
#### South Korea
- https://www.researchgate.net/publication/221911782_Application_of_Artificial_Neural_Network_for_Mineral_Potential_Mapping 
- https://www.researchgate.net/publication/359861043_Rock_Classification_in_a_Vanadiferous_Titanomagnetite_Deposit_Based_on_Supervised_Machine_Learning#fullTextFileContent Rock Classification in a Vanadiferous Titanomagnetite Deposit Based on Supervised Machine Learning
- https://www.researchgate.net/publication/382131746_Domain_Adaptation_from_Drilling_to_Geophysical_Data_for_Mineral_Exploration
#### Phillipines
- https://www.researchgate.net/publication/359632307_A_Geologically_Constrained_Variational_Autoencoder_for_Mineral_Prospectivity_Mapping
- https://www.researchgate.net/publication/263174923_Application_of_Mineral_Exploration_Models_and_GIS_to_Generate_Mineral_Potential_Maps_as_Input_for_Optimum_Land-Use_Planning_in_the_Philippines
- https://www.researchgate.net/publication/267927677_Data-driven_predictive_mapping_of_gold_prospectivity_Baguio_district_Philippines_Application_of_Random_Forests_algorithm
- https://www.researchgate.net/publication/276271833_Data-Driven_Predictive_Modeling_of_Mineral_Prospectivity_Using_Random_Forests_A_Case_Study_in_Catanduanes_Island_Philippines
- https://www.researchgate.net/publication/209803275_Evidential_belief_functions_for_data-driven_geologically_constrained_mapping_of_gold_potential_Baguio_district_Philippines
- https://www.researchgate.net/publication/241001432_Geologically_Constrained_Probabilistic_Mapping_of_Gold_Potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/263724277_Geologically_Constrained_Fuzzy_Mapping_of_Gold_Mineralization_Potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/229641286_Improved_Wildcat_Modelling_of_Mineral_Prospectivity
- https://www.researchgate.net/publication/238447208_Logistic_Regression_for_Geologically_Constrained_Mapping_of_Gold_Potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/248977334_Mineral_imaging_with_Landsat_TM_data_for_hydrothermal_alteration_mapping_in_heavily-vegetated_terrane​​​​​​
- https://www.researchgate.net/publication/356546133_Mineral_Prospectivity_Mapping_via_Gated_Recurrent_Unit_Model
- https://www.researchgate.net/publication/267640864_Random_forest_predictive_modeling_of_mineral_prospectivity_with_small_number_of_prospects_and_data_with_missing_values_in_Abra_Philippines
- https://www.researchgate.net/publication/3931975_Remote_detection_of_vegetation_stress_for_mineral_exploration
- https://www.researchgate.net/publication/263422015_Where_Are_Porphyry_Copper_Deposits_Spatially_Localized_A_Case_Study_in_Benguet_Province_Philippines
- https://www.researchgate.net/publication/233488614_Wildcat_mapping_of_gold_potential_Baguio_District_Philippines
- https://www.researchgate.net/publication/226982180_Weights_of_Evidence_Modeling_of_Mineral_Potential_A_Case_Study_Using_Small_Number_of_Prospects_Abra_Philippines
#### Russia
- https://www.researchgate.net/publication/358431343_Application_of_Maximum_Entropy_for_Mineral_Prospectivity_Mapping_in_Heavily_Vegetated_Areas_of_Greater_Kurile_Chain_with_Landsat_8_Data
- https://www.researchgate.net/publication/354000754_Mineral_Prospectivity_Mapping_for_Forecasting_Gold_Deposits_in_the_Central_Kolyma_Region_North-East_Russia
#### South Africa
- https://www.researchgate.net/publication/359294267_Data-driven_multi-index_overlay_gold_prospectivity_mapping_using_geophysical_and_remote_sensing_datasets
- https://link.springer.com/article/10.1007/s11053-024-10390-w -> Mineral Reconnaissance Through Scientific Consensus: First National Prospectivity Maps for PGE–Ni–Cu–Cr and Witwatersrand-type Au Deposits in South Africa
- https://www.researchgate.net/publication/361526053_Mineral_prospectivity_mapping_of_gold-base_metal_mineralisation_in_the_Sabie-Pilgrim%27s_Rest_area_Mpumalanga_Province_South_Africa
- https://www.researchgate.net/publication/264296137_PREDICTIVE_BEDROCK_AND_MINERAL_PROSPECTIVITY_MAPPING_IN_THE_GIYANI_GREENSTONE_BELT_SOUTH_AFRICA
- https://www.researchgate.net/publication/268196204_Predictive_mapping_of_prospectivity_for_orogenic_gold_Giyani_greenstone_belt_South_Africa
#### Spain
- https://www.researchgate.net/publication/225656353_Deriving_Optimal_Exploration_Target_Zones_on_Mineral_Prospectivity_Maps
- https://www.researchgate.net/publication/222198648_Knowledge-guided_data-driven_evidential_belief_modeling_of_mineral_prospectivity_in_Cabo_de_Gata_SE_Spain
- https://www.researchgate.net/publication/356639977_Machine_learning_models_for_Hg_prospecting_in_the_Almaden_mining_district
- https://www.researchgate.net/publication/43165602_Methodology_for_deriving_optimal_exploration_target_zones
- https://www.researchgate.net/publication/263542579_Optimal_Exploration_Target_Zones
- https://www.researchgate.net/publication/222892103_Optimal_field_sampling_for_targeting_minerals_using_hyperspectral_data
- https://www.researchgate.net/publication/271671416_Predictive_modelling_of_gold_potential_with_the_integration_of_multisource_information_based_on_random_forest_a_case_study_on_the_Rodalquilar_area_Southern_Spain
#### Sudan
- https://link.springer.com/article/10.1007/s11053-024-10387-5 -> Toward Data-Driven Mineral Prospectivity Mapping from Remote Sensing Data Using Deep Forest Predictive Model [UNSEEN]
#### Sweden
- https://www.researchgate.net/publication/259128115_Biogeochemical_expression_of_rare_earth_element_and_zirconium_mineralization_at_Norra_Karr_Southern_Sweden
- https://www.researchgate.net/publication/260086862_COMPARISION_OF_VMS_PROSPECTIVITY_MAPPING_BY_EBF_AND_WOFE_MODELING_THE_SKELLEFTE_DISTRICT_SWEDEN
- https://www.researchgate.net/publication/336086368_GIS-based_mineral_system_approach_for_prospectivity_mapping_of_iron-oxide_apatite-bearing_mineralisation_in_Bergslagen_Sweden
- https://www.researchgate.net/publication/229347041_Predictive_mapping_of_prospectivity_and_quantitative_estimation_of_undiscovered_VMS_deposits_in_Skellefte_district_Sweden
- https://www.researchgate.net/publication/260086947_PRELIMINARY_GIS-BASED_ANALYSIS_OF_REGIONAL-SCALE_VMS_PROSPECTIVITY_IN_THE_SKELLEFTE_REGION_SWEDEN
#### Tanzania
- https://www.sciencedirect.com/science/article/pii/S2666261224000270 -> Machine learning based prospect targeting: A case of gold occurrence in central parts of Tanzania, East Africa
#### Uganda
- https://www.researchgate.net/publication/242339962_Predictive_mapping_for_orogenic_gold_prospectivity_in_Uganda
- https://www.researchgate.net/publication/262566098_Predictive_Mapping_of_Prospectivity_for_Orogenic_Gold_in_Uganda
- https://www.researchgate.net/publication/381219015_Machine_Learning_Application_in_Predictive_Mineral_Mapping_of_Southwestern_Uganda_Leveraging_Airborne_Magnetic_Radiometric_and_Electromagnetic_Data 


#### United Kingdom
- https://www.researchgate.net/publication/383580839_Improved_mineral_prospectivity_mapping_using_graph_neural_networks 

#### USA
- https://www.researchgate.net/publication/338663292_A_Predictive_Geospatial_Exploration_Model_for_Mississippi_Valley_Type_Pb-Zn_Mineralization_in_the_Southeast_Missouri_Lead_District
- Machine Learning and Plate Tectonic Analysis for Mantle Heterogeneity, Paleoclimate, and Critical Minerals -> https://repository.arizona.edu/handle/10150/675507?show=full
- https://www.sciencedirect.com/science/article/abs/pii/S0375674218300396?via%3Dihub -> Machine learning strategies for classification and prediction of alteration facies: Examples from the Rosemont Cu-Mo-Ag skarn deposit, SE Tucson Arizona
 - [presentation of the above!] https://www.slideshare.net/JuanCarlosOrdezCalde/geology-chemostratigraphy-and-alteration-geochemistry-of-the-rosemont-cumoag-skarn-deposit-southern-arizona
 - https://github.com/rohitash-chandra/research/blob/master/presentations/CSIRO%20Minerals-Seminar-September2022.pdf -> Machine Learning for Mineral Exploration: A Data Odyssey
	- Video https://www.youtube.com/watch?v=zhXuPQy7mk8&t=561s -> Talks about using plate subduction and associated statistics via GPlates
#### Zambia
- https://www.researchgate.net/publication/263542565_APPLICATION_OF_REMOTE_SENSING_AND_SPATIAL_DATA_INTEGRATION_TO_PREDICT_POTENTIAL_ZONES_FOR_AQUAMARINE-BEARING_PEGMATITES_LUNDAZI_AREA_NORTHEAST_ZAMBIA
- https://www.researchgate.net/publication/264041472_Geological_and_Mineral_Potential_Mapping_by_Geoscience_Data_Integration
#### Zimbabwe
- https://www.researchgate.net/publication/260792212_Nickel_Sulphide_Deposits_in_Archaean_Greenstone_Belts_in_Zimbabwe_Review_and_Prospectivity_Analysis

## GENERAL PAPERS
 
### Overviews
- https://www.sciencedirect.com/science/article/pii/S2772883824000347 -> A review on the applications of airborne geophysical and remote sensing datasets in epithermal gold mineralisation mapping
- https://www.researchgate.net/publication/353530416_A_Systematic_Review_on_the_Application_of_Machine_Learning_in_Exploiting_Mineralogical_Data_in_Mining_and_Mineral_Industry 
- https://www.researchgate.net/publication/365777421_Computer_Vision_and_Pattern_Recognition_for_the_Analysis_of_2D3D_Remote_Sensing_Data_in_Geoscience_A_Survey - Computer Vision and Pattern Recognition for the Analysis of 2D/3D Remote Sensing Data in Geoscience: A Survey
- https://www.researchgate.net/publication/352104303_Deep_Learning_for_Geophysics_Current_and_Future_Trends
- https://www.proquest.com/openview/e7bec6c8ee50183b5049516b000d4f5c/1?pq-origsite=gscholar&cbl=18750&diss=y -> Probabilistic Knowledge-Guided Machine Learning in Engineering and Geoscience Systems 
 - [KGMLPrescribedFires](https://github.com/sharm636/KGMLPrescribedFires) repository for one paper / part of above dissertation
### Deposits
- https://pubs.er.usgs.gov/publication/ofr20211049 -> Deposit Classification Scheme for the Critical Minerals Mapping Initiative Global Geochemical Database
### ESG
- https://www.escubed.org/journals/earth-science-systems-and-society/articles/10.3389/esss.2024.10109/full -> Geospatial Data and Deep Learning Expose ESG Risks to Critical Raw Materials Supply: The Case of Lithium
### Geochemistry
Causal Discovery and Deep Learning Algorithms for Detecting Geochemical Patterns Associated with Gold-Polymetallic Mineralization: A Case Study of the Edongnan Region
- https://link.springer.com/article/10.1007/s11053-024-10408-3 -> A New Sphalerite Thermometer Based on Machine Learning with Trace Element Geochemistry
- https://www.researchgate.net/publication/378150628_A_SMOTified_extreme_learning_machine_for_identifying_mineralization_anomalies_from_geochemical_exploration_data_a_case_study_from_the_Yeniugou_area_Xinjiang_China A SMOTified extreme learning machine for identifying mineralization anomalies from geochemical exploration data
- https://ui.adsabs.harvard.edu/abs/2018EGUGA..20.4169R/abstract -> Accelerating minerals exploration with in-field characterisation, sample tracking and active machine learning
- https://www.researchgate.net/publication/375509344_Alteration_assemblage_characterization_using_machine_learning_applied_to_high_resolution_drill-core_images_hyperspectral_data_and_geochemistry
- https://qspace.library.queensu.ca/items/38f52d19-609d-4916-bcd0-3ce20675dee3/full - > Application of Computational Methods to Data Integration and Geoscientific Problems in Mineral Exploration and Mining
- https://www.sciencedirect.com/science/article/pii/S0169136822005509?dgcid=rss_sd_all -> Applying neural networks-based modelling to the prediction of mineralization: A case-study using the Western Australian Geochemistry (WACHEM) database
- https://www.sciencedirect.com/science/article/pii/S0169136824002099 -> Development of a machine learning model to classify mineral deposits using sphalerite chemistry and mineral assemblages
- https://www.sciencedirect.com/science/article/pii/S0169136824002403 -> Discrimination of deposit types using magnetite geochemistry based on machine learning
- https://www.researchgate.net/publication/302595237_A_machine_learning_approach_to_geochemical_mapping
- https://www.researchgate.net/publication/369300132_DEEP-LEARNING_IDENTIFICATION_OF_ANOMALOUS_DATA_IN_GEOCHEMICAL_DATASETS_DEEP-LEARNING_IDENTIFICATION_OF_ANOMALOUS_DATA_IN_GEOCHEMICAL_DATASETS
- https://www.researchgate.net/publication/378549920_Denoising_of_geochemical_data_using_deep_learning-Implications_for_regional_surveys -> Denoising of Geochemical Data using Deep Learning–Implications for Regional Surveys]
- https://www.researchgate.net/publication/368489689_Discrimination_of_Pb-Zn_deposit_types_using_sphalerite_geochemistry_New_insights_from_machine_learning_algorithm
- https://www.researchgate.net/publication/381369176_Effectiveness_of_LOF_iForest_and_OCSVM_in_detecting_anomalies_in_stream_sediment_geochemical_data#:~:text=LOF%20outperformed%20iForest%20and%20OCSVM,patterns%20in%20the%20iForest%20map 
- Fusion of Geochemical Data and Remote Sensing
Data Based on Convolutional Neural Network -> https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10758357
- https://dzkjqb.cug.edu.cn/en/article/doi/10.19509/j.cnki.dzkq.tb20220423 -> Gaussian mixture model in geochemical anomaly delineation of stream sediments: A case study of Xupu, Hunan Province [UNSEEN]
- https://www.sciencedirect.com/science/article/pii/S0883292724002427 -> Geologically constrained unsupervised dual-branch deep learning algorithm for geochemical anomalies identification
- https://www.researchgate.net/publication/365953549_Incorporating_the_genetic_and_firefly_optimization_algorithms_into_K-means_clustering_method_for_detection_of_porphyry_and_skarn_Cu-related_geochemical_footprints_in_Baft_district_Kerman_Iran
- https://www.researchgate.net/publication/369768936_Infomax-based_deep_autoencoder_network_for_recognition_of_multi-element_geochemical_anomalies_linked_to_mineralization -> Paywalled
- https://www.sciencedirect.com/science/article/abs/pii/S0098300424001626 -> Local phase-constrained convolutional autoencoder network for identifying multivariate geochemical anomalies
- https://www.researchgate.net/publication/354564681_Machine_Learning_for_Identification_of_Primary_Water_Concentrations_in_Mantle_Pyroxene
- https://www.researchgate.net/publication/366210211_Machine_Learning_Prediction_of_Ore_Deposit_Genetic_Type_Using_Magnetite_Geochemistry 
- https://link.springer.com/article/10.1007/s42461-024-01013-2 -> NIR-Spectroscopy and Machine Learning Models to Pre-concentrate Copper Hosted Within Sedimentary Rocks[UNSEEN]
- https://www.researchsquare.com/article/rs-4106957/v1 -> Multi-element geochemical anomaly recognition applying geologically-constrained convolutional deep learning algorithm with Butterworth filtering
- https://www.researchgate.net/publication/369241349_Quantifying_continental_crust_thickness_using_the_machine_learning_method
- https://link.springer.com/article/10.1007/s11004-024-10158-1 -> Spatial-Spectrum Two-Branch Model Based on a Superpixel Graph Convolutional Network and 1DCNN for Geochemical Anomaly Identification
- https://www.researchgate.net/publication/334651800_Using_machine_learning_to_estimate_a_key_missing_geochemical_variable_in_mining_exploration_Application_of_the_Random_Forest_algorithm_to_multi-sensor_core_logging_data
#### Apatite
- https://www.researchgate.net/publication/377892369_Apatite_trace_element_composition_as_an_indicator_of_ore_deposit_types_A_machine_learning_approachApatite trace element composition as an indicator of ore deposit types: A machine learning approach
- https://www.researchgate.net/publication/369729999_Visual_Interpretation_of_Machine_Learning_Genetical_Classification_of_Apatite_from_Various_Ore_Sources

### Geology
#### Alteration
- https://ieeexplore.ieee.org/abstract/document/10544529 -> Remote sensing data processing using convolutional neural networks for mapping alteration zones [UNSEEN]
#### Depth
- https://www.researchgate.net/publication/332263305_A_speedy_update_on_machine_learning_applied_to_bedrock_mapping_using_geochemistry_or_geophysics_examples_from_the_Pacific_Rim_and_nearby
 - https://eprints.utas.edu.au/32368/ - thesis paper update 
- https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2024.1407173/full -> Deep learning for geological mapping in the overburden area
- https://www.researchgate.net/publication/280038632_Estimating_the_fill_thickness_and_bedrock_topography_in_intermontane_valleys_using_artificial_neural_networks_-_Supporting_Information 
- https://www.researchgate.net/publication/311783770_Mapping_the_global_depth_to_bedrock_for_land_surface_modeling
- https://www.researchgate.net/publication/379813337_Contribution_to_advancing_aquifer_geometric_mapping_using_machine_learning_and_deep_learning_techniques_a_case_study_of_the_AL_Haouz-Mejjate_aquifer_Marrakech_Morocco
- https://www.linkedin.com/pulse/depth-basement-modelling-machine-learning-perspective-n5gyc/?trackingId=qFSktvVPUiSa2V2nlmXVoQ%3D%3D
#### Drill Core
- https://pubmed.ncbi.nlm.nih.gov/35776744/ - Deep learning based lithology classification of drill core images
- https://www.researchgate.net/publication/381445417_Machine_Learning_for_Lithology_Analysis_using_a_Multi-Modal_Approach_of_Integrating_XRF_and_XCT_data 
- https://www.researchgate.net/publication/379760986_A_machine_vision_approach_for_detecting_changes_in_drill_core_textures_using_optical_images
- https://www.sciencedirect.com/science/article/pii/S2949891024002112 -> Sensitivity analysis of similarity learning models for well-intervals based on logging data
- https://www.sciencedirect.com/science/article/pii/S2949891024003828 -> CoreViT: a new vision transformer model for lithology identification in cores
#### General
- https://www.researchgate.net/publication/390084932_A_Deep_Learning_Method_for_3D_Geological_Modeling_Using_ET4DD_with_Offset-Attention_Mechanism -> A deep learning method for 3D geological modeling using ET4DD with offset-attention mechanism [UNSEEN - repo listed in paper but not available]
- https://www.sciencedirect.com/science/article/pii/S0034425724002323 -> Deep learning-based geological map generation using geological routes
- https://www.researchgate.net/publication/354781583_Deep_learning_framework_for_geological_symbol_detection_on_geological_maps
- https://www.researchgate.net/publication/335104674_Does_shallow_geological_knowledge_help_neural-networks_to_predict_deep_units
- https://www.researchgate.net/publication/379939974_Graph_convolutional_network_for_lithological_classification_and_mapping_using_stream_sediment_geochemical_data_and_geophysical_data
- https://www.sciencedirect.com/science/article/abs/pii/S0098300424001493-> FlexLogNet: A flexible deep learning-based well-log completion method of adaptively using what you have to predict what you are missing
- https://ieeexplore.ieee.org/abstract/document/10493129 -> Geological Background Prototype Learning Enhanced Network for Remote Sensing-Based Engineering Geological Lithology Interpretation in Highly Vegetated Areas [Unseen]
- https://www.sciencedirect.com/science/article/pii/S2096249524000619 -> Generating extremely low-dimensional representation of subsurface earth models using vector quantization and deep Autoencoder
- https://www.researchgate.net/publication/370175012_GeoPDNN_A_Semisupervised_Deep_Learning_Neural_Network_Using_Pseudolabels_for_Three-dimensional_Urban_Geological_Modelling_and_Uncertainty_Analysis_from_Borehole_Data
- https://www.researchsquare.com/article/rs-4805227/v1 -> Synergizing AI with Geology: Exploring VisionTransformers for Rock Classification
- https://www.researchgate.net/publication/343511849_Identification_of_intrusive_lithologies_in_volcanic_terrains_in_British_Columbia_by_machine_learning_using_Random_Forests_the_value_of_using_a_soft_classifier
- https://www.sciencedirect.com/science/article/pii/S0169136824000921 -> Machine learning-based field geological mapping: A new exploration of geological survey data acquisition strategy https://www.researchgate.net/publication/324411647_Predicting_rock_type_and_detecting_hydrothermal_alteration_using_machine_learning_and_petrophysical_properties_of_the_Canadian_Malartic_ore_and_host_rocks_Pontiac_Subprovince_Quebec_Canada 
- https://www.sciencedirect.com/science/article/abs/pii/S0895981124001743 -> Utilizing Random Forest algorithm for identifying mafic and ultramafic rocks in the Gameleira Suite, Archean-Paleoproterozoic basement of the Brasília Belt, Brazil
- https://arxiv.org/pdf/2407.18100 -> DINOv2 Rocks Geological Image Analysis: Classification
- https://jgsb.sgb.gov.br/index.php/journal/article/view/252 -> Unveiling geological complexity in the Serra Dourada Granite using self-organizing maps and hierarchical clustering: Insights for REE prospecting in the Goiás Tin Province, Brasília Belt, Central Brazil
#### Geochronology
- https://www.researchgate.net/publication/379077847_Tracing_Andean_Origins_A_Machine_Learning_Framework_for_Lead_Isotopes
- https://www.sciencedirect.com/science/article/pii/S0098300425002468?via%3Dihub#appsec1 -> Raising the bar: Deep learning on comprehensive database sets new benchmark for automated fission-track detection
#### Geomorphology
- https://agu.confex.com/agu/fm18/mediafile/Handout/Paper427843/Landforms%20Poster.pdf -> Using machine learning to classify landforms for minerals exploration
- https://www.tandfonline.com/doi/abs/10.1080/13658816.2024.2414409 -> GeomorPM: a geomorphic pretrained model integrating convolution and Transformer architectures based on DEM data
#### Lithology
- https://www.researchgate.net/publication/389767586_Machine_Learning_for_Characterizing_Magma_Fertility_in_Porphyry_Copper_Deposits_A_Case_Study_of_Southeastern_Tibet
- https://www.sciencedirect.com/science/article/abs/pii/S0926985125000692 -> Identifying ultramafic rocks using artificial neural network method based on aeromagnetic data [UNSEEN]
- https://link.springer.com/article/10.1007/s11053-024-10396-4 -> Interpretable SHAP Model Combining Meta-learning and Vision Transformer for Lithology Classification Using Limited and Unbalanced Drilling Data in Well Logging [UNSEN]
- https://www.nature.com/articles/s41598-024-66199-3 -> Machine learning and remote sensing-based lithological mapping of the Duwi Shear-Belt area, Central Eastern Desert, Egypt
- https://link.springer.com/article/10.1007/s11053-024-10375-9 - SsL-VGMM: A Semisupervised Machine Learning Model of Multisource Data Fusion for Lithology Prediction [UNSEEN]
- https://www.researchgate.net/publication/380719080_An_integrated_machine_learning_framework_with_uncertainty_quantification_for_three-dimensional_lithological_modeling_from_multi-source_geophysical_data_and_drilling_data
- https://www.bio-conferences.org/articles/bioconf/pdf/2024/34/bioconf_rena23_01005.pdf -> Lithological Mapping using Artificial Intelligence and Remote Sensing data: A Case Study of Bab Boudir region Morocco
#### Mineralogy
- https://pubs.geoscienceworld.org/msa/ammin/article-abstract/doi/10.2138/am-2023-9092/636861/The-application-of-transfer-learning-in-optical -> The application of “transfer learning” in optical microscopy: the petrographic classification of metallic minerals
- https://www.researchgate.net/publication/385074584_Deep_Learning-Based_Mineral_Classification_Using_Pre-Trained_VGG16_Model_with_Data_Augmentation_Challenges_and_Future_Directions
#### Resource
- https://link.springer.com/article/10.1007/s11053-025-10485-y -> Uncertainty Quantification of Microblock-Based Resource Models and Sequencing of Sampling
#### Stratigraphy
- https://www.researchgate.net/publication/335486001_A_Stratigraphic_Prediction_Method_Based_on_Machine_Learning
- https://www.researchgate.net/publication/346641320_Classifying_basin-scale_stratigraphic_geometries_from_subsurface_formation_tops_with_machine_learning
#### Structure
- https://www.sciencedirect.com/science/article/pii/S0098300421000285 -> A machine learning model for structural trend fields
- https://onlinelibrary.wiley.com/doi/full/10.1111/1365-2478.13589 -> Inferring fault structures and overburden depth in 3D from geophysical data using machine learning algorithms – A case study on the Fenelon gold deposit, Quebec, Canada
- https://www.sciencedirect.com/science/article/pii/S019181412400138X -> Mapping paleostress trajectories by means of the clustering of reduced stress tensors determined from homogeneous and heterogeneous data sets
- https://www.researchgate.net/publication/332267249_Seismic_fault_detection_using_an_encoder-decoder_convolutional_neural_network_with_a_small_training_set 
- https://www.researchgate.net/publication/377168034_Unsupervised_machine_learning_and_depth_clusters_of_Euler_deconvolution_of_magnetic_data_a_new_approach_to_imaging_geological_structures
- https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggae226/7701418 -> Use of Decision Tree Ensembles for Crustal Structure Imaging from Receiver Functions
#### Tectonics
- https://www.researchgate.net/publication/371594975_Assessing_plate_reconstruction_models_using_plate_driving_force_consistency_tests
- https://www.researchgate.net/publication/333182666_Decoding_Earth's_plate_tectonic_history_using_sparse_geochemical_data
- https://www.researchgate.net/publication/376519740_Machine_learning_and_tectonic_setting_determination_Bridging_the_gap_between_Earth_scientists_and_data_scientists
- https://pubs.geoscienceworld.org/gsa/geology/article-abstract/doi/10.1130/G52466.1/648458/Prediction-of-CO2-content-in-mid-ocean-ridge -> Prediction of CO2 content in mid-ocean ridge basalts via a machine learning approach

### Geophysics
#### Foundation
- https://www.researchgate.net/publication/373714604_Seismic_Foundation_Model_SFM_a_new_generation_deep_learning_model_in_geophysics
#### General
- https://essopenarchive.org/users/841077/articles/1231187-bayesian-inference-in-geophysics-with-ai-enhanced-markov-chain-monte-carlo -> Bayesian Inference in Geophysics with AI-enhanced Markov chain Monte Carlo
- https://www.researchgate.net/publication/353789276_Geology_differentiation_by_applying_unsupervised_machine_learning_to_multiple_independent_geophysical_inversions
- https://www.sciencedirect.com/science/article/pii/S001379522100137X - Joint interpretation of geophysical data: Applying machine learning to the modeling of an evaporitic sequence in Villar de Cañas (Spain)
- https://www.sciencedirect.com/science/article/pii/S2666544121000253 - Microleveling aerogeophysical data using deep convolutional network and MoG-RPCA
- https://www.researchgate.net/publication/368550674_Objective_classification_of_high-resolution_geophysical_data_Empowering_the_next_generation_of_mineral_exploration_in_Sierra_Leone
- https://datarock.com.au/blog/transfer-learning-with-seismic-attributes -> Transfer Learning with Seismic Attributes
#### Potential Fields
- https://api.research-repository.uwa.edu.au/ws/portalfiles/portal/390212334/THESIS_-_DOCTOR_OF_PHILOSOPHY_-_SMITH_Luke_Thomas_-_2023_.pdf -> Potential Field Geophysics Enhancement Using Conteporary Deep Learning
- https://ieeexplore.ieee.org/abstract/document/10767251 -> A Stable Method for Estimating the Derivatives of Potential Field Data Based on Deep Learning ->  [UNSEEN]
https://www.researchgate.net/publication/389575997_Pole_Transformation_of_Magnetic_Data_Using_CNN-Based_Deep_Learning_Models
- https://www.mdpi.com/2073-8994/17/4/523 -> Inversion of Gravity Anomalies Based on U-Net Network
#### EM
- https://d197for5662m48.cloudfront.net/documents/publicationstatus/206704/preprint_pdf/59681a0a2c571bc2a9006f37517bc6ef.pdf -> A Fast Three-dimensional Imaging Scheme of Airborne Time Domain Electromagnetic Data using Deep Learning
- https://www.researchgate.net/publication/351507441_A_Neural_Network-Based_Hybrid_Framework_for_Least-Squares_Inversion_of_Transient_Electromagnetic_Data
- https://www.researchgate.net/profile/Yunhe-Liu/publication/382196526_An_Efficient_Bayesian_Inference_for_Geo-electromagnetic_Data_Inversion_based_on_Surrogate_Modeling_with_Adaptive_Sampling_DNN
- https://www.researchgate.net/publication/325980016_Agglomerative_hierarchical_clustering_of_airborne_electromagnetic_data_for_multi-scale_geological_studies
- https://www.earthdoc.org/content/papers/10.3997/2214-4609.202410980 -> Deep Learning Assisted 2-D Current Density Modelling of Very Low Frequency Electromagnetic Data
- https://npg.copernicus.org/articles/26/13/2019/ -> Denoising stacked autoencoders for transient electromagnetic signal denoising
- https://www.researchgate.net/publication/373836226_An_information_theoretic_Bayesian_uncertainty_analysis_of_AEM_systems_over_Menindee_Lake_Australia -> An information theoretic Bayesian uncertainty analysis of AEM systems over Menindee Lake, Australia
- https://www.researchgate.net/publication/348850484_Effect_of_Data_Normalization_on_Neural_Networks_for_the_Forward_Modelling_of_Transient_Electromagnetic_Data
- https://www.researchgate.net/publication/342153377_Fast_imaging_of_time-domain_airborne_EM_data_using_deep_learning_technology
- https://library.seg.org/doi/10.4133/JEEG4.2.93 -> Neural Network Interpretation of High Frequency Electromagnetic Ellipticity Data Part I: Understanding the Half-Space and Layered Earth Response
- https://arxiv.org/abs/2207.12607 -> Physics Embedded Machine Learning for Electromagnetic Data Imaging
- https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggae244/7713480 -> Physics-guided deep learning-based inversion for airborne electromagnetic data
- https://library.seg.org/doi/abs/10.1190/geo2024-0282.1 -> Comparative Analysis of Deep Learning and Traditional Airborne Electromagnetic Data Processing: A Case Study [UNSEEN]
- https://www.researchgate.net/publication/359441000_Surface_parameters_and_bedrock_properties_covary_across_a_mountainous_watershed_Insights_from_machine_learning_and_geophysics
- https://www.researchgate.net/publication/337166479_Using_machine_learning_to_interpret_3D_airborne_electromagnetic_inversions
- https://www.researchgate.net/publication/344397798_TEMDnet_A_Novel_Deep_Denoising_Network_for_Transient_Electromagnetic_Signal_With_Signal-to-Image_Transformation
- https://www.researchgate.net/publication/366391168_Two-dimensional_fast_imaging_of_airborne_EM_data_based_on_U-net
#### ERT
- https://www.sciencedirect.com/science/article/pii/S0013795224001893 -> Geo-constrained clustering of resistivity data revealing the heterogeneous lithological architectures and the distinctive geoelectrical signature of shallow deposits
#### Gravity
- https://ieeexplore.ieee.org/abstract/document/10597585 -> 3D Basement Relief and Density Inversion Based on EfficientNetV2 Deep Learning Network [UNSEEN]
- https://link.springer.com/article/10.1007/s11770-024-1096-5 -> 3D gravity inversion using cycle-consistent generative adversarial network [UNSEEN]
- https://www.researchgate.net/publication/365142017_3D_gravity_inversion_based_on_deep_learning
- https://www.researchgate.net/publication/378930477_A_Deep_Learning_Gravity_Inversion_Method_Based_on_a_Self-Constrained_Network_and_Its_Application
- https://www.researchgate.net/publication/362276214_DecNet_Decomposition_network_for_3D_gravity_inversion -> Olympic Dam example here
- https://www.researchgate.net/publication/368448190_Deep_Learning_to_estimate_the_basement_depth_by_gravity_data_using_Feedforward_neural_network
- https://www.researchgate.net/publication/326231731_Depth_and_Lineament_Maps_Derived_from_North_Cameroon_Gravity_Data_Computed_by_Artificial_Neural_Network_International_Journal_of_Geophysics_vol_2018_Article_ID_1298087_13_pages_2018
- https://www.researchgate.net/publication/366922016_Fast_imaging_for_the_3D_density_structures_by_machine_learning_approach
- https://www.researchgate.net/publication/370230217_Inversion_of_the_Gravity_Gradiometry_Data_by_ResUet_Network_An_Application_in_Nordkapp_Basin_Barents_Sea
- https://library.seg.org/doi/abs/10.1190/geo2024-0150.1 -> Integration of PSPU-Net gravity inversion neural network with gravity data for enhanced 3D basement relief estimation
- https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2022.897055/full -> High-precision downward continuation of the potential field based on the D-Unet network
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10672527 -> RTM Gravity Forward Modeling Using Improved Fully Connected Deep Neural Networks
#### Hyperspectral
- https://www.researchgate.net/publication/380391736_A_review_on_hyperspectral_imagery_application_for_lithological_mapping_and_mineral_prospecting_Machine_learning_techniques_and_future_prospects 
- https://www.researchgate.net/publication/372876863_Ore-Grade_Estimation_from_Hyperspectral_Data_Using_Convolutional_Neural_Networks_A_Case_Study_at_the_Olympic_Dam_Iron_Oxide_Copper-Gold_Deposit_Australia [UNSEEN]
#### Joint Inversion
- https://www.researchgate.net/publication/383454185_Deep_joint_inversion_of_electromagnetic_seismic_and_gravity_data
- https://ieeexplore.ieee.org/abstract/document/10677418 -> Joint Inversion of Seismic and Resistivity Data Powered by Deep-learning [UNSEEN]
- Deep learning-based geophysical joint inversion using partial channel drop method -> https://www.sciencedirect.com/science/article/abs/pii/S0926985124002702
- https://www.mdpi.com/2076-3417/15/6/3125 -> A Hybrid Deep Learning Approach for Integrating Transient Electromagnetic and Magnetic Data to Enhance Subsurface Anomaly Detection
### Magnetics
- https://www.researchgate.net/publication/348697645_3D_geological_structure_inversion_from_Noddy-generated_magnetic_data_using_deep_learning_methods
- https://www.researchgate.net/publication/360288249_3D_Inversion_of_Magnetic_Gradient_Tensor_Data_Based_on_Convolutional_Neural_Networks
- https://www.researchgate.net/publication/295902270_Artificial_neural_network_inversion_of_magnetic_anomalies_caused_by_2D_fault_structures
- https://www.researchgate.net/publication/354002966_Convolutional_neural_networks_for_the_characterization_of_magnetic_anomalies
- https://www.researchgate.net/publication/354772176_Convolution_Neural_Networks_Applied_to_the_Interpretation_of_Lineaments_in_Aeromagnetic_Data
- https://www.researchgate.net/publication/363550362_High-precision_downward_continuation_of_the_potential_field_based_on_the_D-Unet_network
- https://www.sciencedirect.com/science/article/pii/S0169136822004279?via%3Dihub -> Magnetic grid resolution enhancement using machine learning: A case study from the Eastern Goldfields Superterrane
- https://www.researchgate.net/publication/347173621_Predicting_Magnetization_Directions_Using_Convolutional_Neural_Networks -> Paywalled
- https://www.researchgate.net/publication/361114986_Reseaux_de_Neurones_Convolutifs_pour_la_Caracterisation_d'Anomalies_Magnetiques -> French original of the above
#### Magnetotellurics
- https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-024-03842-3 -> 2D magnetotelluric imaging method based on visionary self-attention mechanism and data science
- https://ieeexplore.ieee.org/abstract/document/10955415- -> 3DInception-U: Lightweight Network for 3-D Magnetotelluric Inversion Based on Inception Module [UNSEE]
- https://ieeexplore.ieee.org/abstract/document/10530937 -> A Magnetotelluric Data Denoising Method Based on Lightweight Ensemble Learning [UNSEEN]
- https://academic.oup.com/gji/advance-article/doi/10.1093/gji/ggae166/7674890 -> Deep basin conductor characterization using machine learning-assisted magnetotelluric Bayesian inversion in the SW Barents Sea
- http://en.dzkx.org/article/doi/10.6038/cjg2024R0580 -> Fast inversion method of apparent resistivity based on deep learning
- https://www.researchgate.net/publication/367504269_Flexible_and_accurate_prior_model_construction_based_on_deep_learning_for_2D_magnetotelluric_data_inversion
- https://www.sciencedirect.com/science/article/pii/S2214579624000510 -> Intelligent Geological Interpretation of AMT Data Based on Machine Learning
- https://ieeexplore.ieee.org/abstract/document/10551853 -> Magnetotelluric Data Inversion Based on Deep Learning with the Self-attention Mechanism
- https://www.researchgate.net/publication/361741409_Physics-Driven_Deep_Learning_Inversion_with_Application_to_Magnetotelluric
- https://www.researchgate.net/publication/355568465_Stochastic_inversion_of_magnetotelluric_data_using_deep_reinforcement_learning
- https://www.researchgate.net/publication/354360079_Two-dimensional_deep_learning_inversion_of_magnetotelluric_sounding_data
- https://ieeexplore.ieee.org/abstract/document/10530923 -> Three Dimensional Magnetotelluric Forward Modeling Through Deep Learning
#### Passive Seismic
- https://nature.com/articles/s41467-020-17841-x -> Clustering earthquake signals and background noises in continuous seismic data with unsupervised deep learning
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022GL099053 -> Eikonal Tomography With Physics-Informed Neural Networks: Rayleigh Wave Phase Velocity in the Northeastern Margin of the Tibetan Plateau
- https://arxiv.org/abs/2403.15095 -> End-to-End Mineral Exploration with Artificial Intelligence and Ambient Noise Tomography
- https://www.nature.com/articles/s41598-019-50381-z -> High-resolution seismic tomography of Long Beach, CA using machine learning
#### Seismic
- https://www.sciencedirect.com/science/article/pii/S0040195124002166 -> Reprocessing and interpretation of legacy seismic data using machine learning from the Granada Basin, Spain
- https://ojs.uni-miskolc.hu/index.php/geosciences/article/view/3313 -> EDGE DETECTION OF TOMOGRAPHIC IMAGES USING TRADITIONAL AND DEEP LEARNING TOOLS
- https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024JH000432 -> One‐Fit‐All Transformer for Multimodal Geophysical Inversion: Method and Application
#### Surface Resistivity
- https://www.researchgate.net/publication/367606119_Deriving_Surface_Resistivity_from_Polarimetric_SAR_Data_Using_Dual-Input_UNet
#### Uncertainty
- https://library.seg.org/doi/abs/10.1190/GEM2024-084.1 -> Quantifying uncertainty in 3D geophysical inverse problems: Advancing from deterministic to Bayesian and deep generative models [UNSEEN]

#### Geothermal
- https://www.osti.gov/biblio/2335471 - Applications of Machine Learning Techniques to Geothermal Play Fairway Analysis in the Great Basin Region, Nevada [adjacent but interesting]
 - https://gdr.openei.org/submissions/1402 - Associated code
 - https://catalog.data.gov/dataset/python-codebase-and-jupyter-notebooks-applications-of-machine-learning-techniques-to-geoth 
 - https://www.researchgate.net/publication/341418586_Preliminary_Report_on_Applications_of_Machine_Learning_Techniques_to_the_Nevada_Geothermal_Play_Fairway_Analysis
 
### Maps
- https://www.researchgate.net/publication/347786302_Semantic_Segmentation_Deep_Learning_for_Extracting_Surface_Mine_Extents_from_Historic_Topographic_Maps
### Mineral
- https://www.researchgate.net/publication/357942198_Mineral_classification_of_lithium-bearing_pegmatites_based_on_laser-induced_breakdown_spectroscopy_Application_of_semi-supervised_learning_to_detect_known_minerals_and_unknown_material
- https://iopscience.iop.org/article/10.1088/1755-1315/1032/1/012046 -> Classifying Minerals using Deep Learning Algorithms
- https://www.researchgate.net/publication/370835450_Predicting_new_mineral_occurrences_and_planetary_analog_environments_via_mineral_association_analysis
- https://www.researchgate.net/publication/361230503_What_is_Mineral_Informatics
### NLP
- https://www.frontiersin.org/journals/earth-science/articles/10.3389/feart.2025.1530004/full#B23 -> Assessing named entity recognition by using geoscience domain schemas: the case of mineral systems
- https://www.researchgate.net/publication/358616133_Chinese_Named_Entity_Recognition_in_the_Geoscience_Domain_Based_on_BERT
- https://www.researchgate.net/publication/339394395_Dictionary-Based_Automated_Information_Extraction_From_Geological_Documents_Using_a_Deep_Learning_Algorithm
- https://www.aclweb.org/anthology/2020.lrec-1.568/ -> Embeddings for Named Entity Recognition in Geoscience Portuguese Literature
- https://www.researchgate.net/publication/359186219_Few-shot_learning_for_name_entity_recognition_in_geological_text_based_on_GeoBERT
- https://www.researchgate.net/publication/333464862_GeoDocA_-_Fast_Analysis_of_Geological_Content_in_Mineral_Exploration_Reports_A_Text_Mining_Approach
- https://www.researchgate.net/publication/366710921_Geological_profile-text_information_association_model_of_mineral_exploration_reports_for_fast_analysis_of_geological_content
- https://www.researchgate.net/publication/330835955_Geoscience_Keyphrase_Extraction_Algorithm_Using_Enhanced_Word_Embedding [UNSEEN]
- https://www.researchgate.net/publication/332997161_GNER_A_Generative_Model_for_Geological_Named_Entity_Recognition_Without_Labeled_Data_Using_Deep_Learning
- https://www.researchgate.net/publication/321850315_Information_extraction_and_knowledge_graph_construction_from_geoscience_literature
- https://www.researchgate.net/publication/365929623_Named_Entity_Annotation_Schema_for_Geological_Literature_Mining_in_the_Domain_of_Porphyry_Copper_Deposits
- https://www.researchgate.net/publication/329621358_Ontology-Based_Enhanced_Word_Embedding_for_Automated_Information_Extraction_from_Geoscience_Reports
- https://www.researchgate.net/publication/379808469_Ontology-driven_relational_data_mapping_for_constructing_a_knowledge_graph_of_porphyry_copper_deposits -> Ontology-driven relational data mapping for constructing a knowledge graph of porphyry copper deposits
- https://www.researchgate.net/publication/327709479_Prospecting_Information_Extraction_by_Text_Mining_Based_on_Convolutional_Neural_Networks-A_Case_Study_of_the_Lala_Copper_Deposit_China
- https://ieeexplore.ieee.org/document/8711400 -> Research and Application on Geoscience Literature Knowledge Discovery Technology
- https://www.researchgate.net/publication/332328315_Text_Mining_to_Facilitate_Domain_Knowledge_Discovery
- https://www.researchgate.net/publication/351238658_Understanding_Ore-Forming_Conditions_using_Machine_Reading_of_Text
- https://www.researchgate.net/publication/359089763_Visual_analytics_and_information_extraction_of_geological_content_for_text-based_mineral_exploration_reports
- https://www.researchgate.net/publication/354754114_What_is_this_article_about_Generative_summarization_with_the_BERT_model_in_the_geosciences_domain
- https://www.slideshare.net/phcleverley/where-text-analytics-meets-geoscience -> Where text analytics meets geoscience

### Petrography
- https://www.researchgate.net/publication/335226326_Digital_petrography_Mineralogy_and_porosity_identification_using_machine_learning_algorithms_in_petrographic_thin_section_images

Last edited: 29/09/2020
The below are a collection of works from when I was doing a review
## Public Mineral Prospectivity Mapping

## Overview
- https://www.researchgate.net/publication/331852267_Applying_Spatial_Prospectivity_Mapping_to_Exploration_Targeting_Fundamental_Practical_issues_and_Suggested_Solutions_for_the_Future
- https://www.researchgate.net/publication/284890591_Geochemical_Anomaly_and_Mineral_Prospectivity_Mapping_in_GIS
- https://www.researchgate.net/publication/341472154_Geodata_Science-Based_Mineral_Prospectivity_Mapping_A_Review
- https://www.researchgate.net/publication/275338029_Introduction_to_the_Special_Issue_GIS-based_mineral_potential_modelling_and_geological_data_analyses_for_mineral_exploration
- https://www.researchgate.net/publication/339074334_Introduction_to_the_special_issue_on_spatial_modelling_and_analysis_of_ore-forming_processes_in_mineral_exploration_targeting
- https://www.researchgate.net/publication/317319129_Natural_Resources_Research_Publications_on_Geochemical_Anomaly_and_Mineral_Potential_Mapping_and_Introduction_to_the_Special_Issue_of_Papers_in_These_Fields
- https://www.researchgate.net/publication/46696293_Selection_of_coherent_deposit-type_locations_and_their_application_in_data-driven_mineral_prospectivity_mapping
## Geochemistry
- https://www.researchgate.net/publication/375926319_A_paradigm_shift_in_Precambrian_research_driven_by_big_data
- https://www.researchgate.net/publication/359447201_A_review_of_machine_learning_in_geochemistry_and_cosmochemistry_Method_improvements_and_applications 
  - https://jaywen.com/files/He_2022_Applied_Geochemistry.pdf
- https://www.researchgate.net/publication/220164381_Application_of_geochemical_zonality_coefficients_in_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/238505045_Analysis_and_mapping_of_geochemical_anomalies_using_logratio-transformed_stream_sediment_data_with_censored_values
- https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2022EA002626 -> Comparative Study on Three Autoencoder-Based Deep Learning Algorithms for Geochemical Anomaly Identification
- https://www.researchgate.net/publication/373758047_Decision-making_within_geochemical_exploration_data_based_on_spatial_uncertainty_-A_new_insight_and_a_futuristic_review
- https://www.researchgate.net/publication/331505001_Deep_learning_and_its_application_in_geochemical_mapping
- https://www.researchgate.net/publication/380262759_Factor_analysis_in_residual_soils_of_the_Iberian_Pyrite_Belt_Spain_Comparison_between_raw_data_log_transformation_data_and_compositional_data [UNSEEN]
- https://www.researchgate.net/publication/272091723_Geochemical_characteristics_of_mineral_deposits_Implications_for_ore_genesis

- https://www.researchgate.net/publication/257189047_Geochemical_mineralization_probability_index_GMPI_A_new_approach_to_generate_enhanced_stream_sediment_geochemical_evidential_map_for_increasing_probability_of_success_in_mineral_potential_mapping
- https://www.researchgate.net/publication/333497470_Integration_of_auto-encoder_network_with_density-based_spatial_clustering_for_geochemical_anomaly_detection_for_mineral_exploration
- https://www.researchgate.net/publication/319303831_Introduction_to_the_thematic_issue_Analysis_of_exploration_geochemical_data_for_mapping_of_anomalies
- https://www.researchgate.net/publication/356722687_Machine_learning-based_prediction_of_trace_element_concentrations_using_data_from_the_Karoo_large_igneous_province_and_its_application_in_prospectivity_mapping#fullTextFileContent
- https://www.degruyter.com/document/doi/10.2138/am-2023-9115/html -> Machine learning applied to apatite compositions for determining mineralization potential [UNSEEN]
- https://www.researchgate.net/publication/257026525_Primary_geochemical_characteristics_of_mineral_deposits_-_Implications_for_exploration
- https://www.researchgate.net/publication/283554338_Recognition_of_geochemical_anomalies_using_a_deep_autoencoder_network
  - https://zarmesh.com/wp-content/uploads/2017/04/Recognition-of-geochemical-anomalies-using-a-deep-autoencoder-network.pdf
- https://www.researchgate.net/publication/349606557_Robust_Feature_Extraction_for_Geochemical_Anomaly_Recognition_Using_a_Stacked_Convolutional_Denoising_Autoencoder [UNSEEN]
- https://www.researchgate.net/publication/375911531_Spatial_Interpolation_Using_Machine_Learning_From_Patterns_and_Regularities_to_Block_Models#fullTextFileContent
- https://www.researchgate.net/publication/259716832_Supervised_and_unsupervised_classification_of_near-mine_soil_Geochemistry_and_Geophysics_data
- https://www.researchgate.net/publication/277813662_Supervised_Geochemical_Anomaly_Detection_by_Pattern_Recognition
- https://www.researchgate.net/publication/249544991_Usefulness_of_stream_order_to_detect_stream_sediment_geochemical_anomalies
- https://www.researchgate.net/publication/321275541_Weighting_stream_sediment_geochemical_samples_as_exploration_indicator_of_deposit_-_type
## Fuzzy
- https://www.researchgate.net/publication/272170968_A_Comparative_Analysis_of_Weights_of_Evidence_Evidential_Belief_Functions_and_Fuzzy_Logic_for_Mineral_Potential_Mapping_Using_Incomplete_Data_at_the_Scale_of_Investigation
- https://www.researchgate.net/publication/267816279_Fuzzification_of_continuous-value_spatial_evidence_for_mineral_prospectivity_mapping
- https://www.researchgate.net/publication/301635716_Union_score_and_fuzzy_logic_mineral_prospectivity_mapping_using_discretized_and_continuous_spatial_evidence_values
## Uncertainty
- https://deliverypdf.ssrn.com/delivery.php?ID=555064031119110002088087068121000096050036019060022069010050000053011056029076002067121000064004002088113115000107115017083105004026015092089005123065040099024112018026013043065104094012124120126039100033055018066074125089104115090100009064122122019003015085069021024027072126106082092110&EXT=pdf&INDEX=TRUE -> Estimating uncertainties in 3-D models of complex fold-and-thrust 2 belts: a case study of the Eastern Alps triangle zone
- https://www.researchgate.net/publication/333339659_Incorporating_conceptual_and_interpretation_uncertainty_to_mineral_prospectivity_modelling
- https://www.researchgate.net/publication/235443307_Managing_uncertainty_in_exploration_targeting
- https://www.researchgate.net/publication/255909185_The_upside_of_uncertainty_Identification_of_lithology_contact_zones_from_airborne_geophysics_and_satellite_data_using_random_forests_and_support_vector_machines
 
## Geospatial Maps
### Australia
- https://www.researchgate.net/publication/334440382_Mapping_iron_oxide_Cu-Au_IOCG_mineral_potential_in_Australia_using_a_knowledge-driven_mineral_systems-based_approach
#### South Australia
- https://www.researchgate.net/publication/335313790_Prospectivity_modelling_of_the_Olympic_Cu-Au_Province - https://services.sarig.sa.gov.au/raster/ProspectivityModelling/wms?service=wms&version=1.1.1&REQUEST=GetCapabilities
- An assessment of the uranium and geothermal prospectivity of east-central South Australia - https://d28rz98at9flks.cloudfront.net/72666/Rec2011_034.pdf
#### NT
- https://www.researchgate.net/publication/285235798_An_assessment_of_the_uranium_and_geothermal_prospectivity_of_the_southern_Northern_Territory
#### WA
- https://www.researchgate.net/publication/273073675_Building_a_machine_learning_classifier_for_iron_ore_prospectivity_in_the_Yilgarn_Craton
- http://dmpbookshop.eruditetechnologies.com.au/product/district-scale-targeting-for-gold-in-the-yilgarn-craton-part-2-of-the-yilgarn-gold-exploration-targeting-atlas.do$55 purchase
- http://dmpbookshop.eruditetechnologies.com.au/product/mineral-prospectivity-of-the-king-leopold-orogen-and-lennard-shelf-analysis-of-potential-field-data-in-the-west-kimberley-region-geographical-product-n14bnzp.do
- http://dmpbookshop.eruditetechnologies.com.au/product/mineral-systems-analysis-of-the-west-musgrave-province-regional-structure-and-prospectivity-modelling-geographical-product-n12dzp.do
- http://dmpbookshop.eruditetechnologies.com.au/product/mineral-systems-analysis-of-the-west-musgrave-province-regional-structure-and-prospectivity-modelling.do  $22 purchase
- https://researchdata.edu.au/predictive-mineral-discovery-gold-mineral/1209568?source=suggested_datasets - Predictive mineral discovery in the eastern Yilgarn Craton: an example of district-scale targeting of an orogenic gold mineral system - https://d28rz98at9flks.cloudfront.net/82617/Y4_Gold_Targeting.zip
- http://dmpbookshop.eruditetechnologies.com.au/product/prospectivity-analysis-of-the-halls-creek-orogen-western-australia-using-a-mineral-systems-approach-geographical-product-n15af3zp.do
- https://researchdata.edu.au/prospectivity-analysis-using-063-m436/1424743 - Prospectivity analysis using a mineral systems approach - Capricorn case study project CSIRO Prospectivity analysis using a mineral systems approach - Capricorn case study project (13.5 GB Download)
- http://dmpbookshop.eruditetechnologies.com.au/product/regional-scale-targeting-for-gold-in-the-yilgarn-craton-part-1-of-the-yilgarn-gold-exploration-targeting-atlas.do $55 purchase
- https://www.researchgate.net/publication/263928515_Towards_Australian_metallogenic_maps_through_space_and_time
- https://www.sciencedirect.com/science/article/abs/pii/S0301926810002111 - Yilgarn
### Brazil
- https://www.researchgate.net/publication/340633563_CATALOG_OF_PROSPECTIVITY_MAPS_OF_SELECTED_AREAS_FROM_BRAZIL
- https://www.researchgate.net/publication/341936771_Modeling_of_Cu-Au_Prospectivity_in_the_Carajas_mineral_province_Brazil_through_Machine_Learning_Dealing_with_Imbalanced_Training_Data
- https://www.researchgate.net/publication/287270273_Nickel_prospective_modelling_using_fuzzy_logic_on_nova_Brasilandia_metasedimentary_belt_Rondonia_Brazil
- https://www.scielo.br/scielo.php?script=sci_arttext&pid=S2317-48892016000200261 - Sao Francisco Craton Nickel
### Australia
- https://www.researchgate.net/publication/248211737_A_continent-wide_study_of_Australia's_uranium_potential
- https://www.researchgate.net/publication/334440382_Mapping_iron_oxide_Cu-Au_IOCG_mineral_potential_in_Australia_using_a_knowledge-driven_mineral_systems-based_approach
- https://researchdata.edu.au/predictive-model-opal-mining-approach/673159/?refer_q=rows=15/sort=score%20desc/class=collection/p=2/q=mineral%20prospectivity%20map/ - Opal
### SA
- https://data.gov.au/dataset/ds-ga-a8619169-1c2a-6697-e044-00144fdd4fa6/details?q= -> An assessment of the uranium and geothermal prospectivity of east central South Australia
- https://d28rz98at9flks.cloudfront.net/72666/Rec2011_034.pdf -> An assessment of the uranium and geothermal prospectivity of east-central South Australia
- https://www.pir.sa.gov.au/__data/assets/pdf_file/0011/239636/204581-001_wise_high.pdf - Eastern Gawler - WPA
- http://www.energymining.sa.gov.au/minerals/knowledge_centre/mesa_journal/previous_feature_articles/new_prospectivity_map
- https://catalog.sarig.sa.gov.au/geonetwork/srv/eng/catalog.search#/metadata/e59cd4ba-1a0a-4911-9e6a-58d80576678d - Olympic Domain IOCG Prospectivity model
- https://www.researchgate.net/publication/335313790_Prospectivity_modelling_of_the_Olympic_Cu-Au_Province - https://services.sarig.sa.gov.au/raster/ProspectivityModelling/wms?service=wms&version=1.1.1&REQUEST=GetCapabilities
### WA
- https://www.sciencedirect.com/science/article/abs/pii/S0301926810002111 - Yilgarn Karol Czarnota
- https://www.researchgate.net/publication/229333177_Prospectivity_analysis_of_the_Plutonic_Marymia_Greenstone_Belt_Western_Australia
- https://www.researchgate.net/publication/280039091_Mineral_systems_approach_applied_to_GIS-based_2D-prospectivity_modelling_of_geological_regions_Insights_from_Western_Australia
- https://www.researchgate.net/publication/351238658_Understanding_Ore-Forming_Conditions_using_Machine_Reading_of_Text
### NT
- https://www.researchgate.net/publication/285235798_An_assessment_of_the_uranium_and_geothermal_prospectivity_of_the_southern_Northern_Territory
- https://www.researchgate.net/publication/342352173_Modelling_gold_potential_in_the_Granites-Tanami_Orogen_NT_Australia_A_comparative_study_using_continuous_and_data-driven_techniques
### NSW
- https://www.resourcesandgeoscience.nsw.gov.au/miners-and-explorers/geoscience-information/projects/mineral-potential-mapping#_southern-_new-_england-_orogen-mineral-potential
- https://www.smedg.org.au/GSNSW_2019_Blevin.pdf - Eastern Lachlan Orogen 
- https://www.researchgate.net/publication/265915602_Comparing_prospectivity_modelling_results_and_past_exploration_data_A_case_study_of_porphyry_Cu-Au_mineral_systems_in_the_Macquarie_Arc_Lachlan_Fold_Belt_New_South_Wales
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
- https://www.researchgate.net/publication/252707107_GIS-based_epithermal_copper_prospectivity_mapping_of_the_Mt_Isa_Inlier_Australia_Implications_for_exploration_targeting
- https://www.researchgate.net/publication/222211452_Predictive_modelling_of_prospectivity_for_Pb-Zn_deposits_in_the_Lawn_Hill_Region_Queensland_Australia
#### New South Wales
- https://www.researchgate.net/publication/336349643_MINERAL_POTENTIAL_MAPPING_AS_A_STRATEGIC_PLANNING_TOOL_IN_THE_EASTERN_LACHLAN_OROGEN_NSW
- https://www.publish.csiro.au/ex/pdf/ASEG2013ab236 - Mineral prospectivity analysis of the Wagga–Omeo belt in NSW
- https://www.researchgate.net/publication/329761040_NSW_Zone_54_Mineral_Systems_Mineral_Potential_Report
- https://www.researchgate.net/publication/337569823_Practical_Implementation_of_Random_Forest-Based_Mineral_Potential_Mapping_for_Porphyry_Cu-Au_Mineralization_in_the_Eastern_Lachlan_Orogen_NSW_Australia
- https://www.researchgate.net/publication/333551776_Translating_expressions_of_intrusion-related_mineral_systems_into_mappable_spatial_proxies_for_mineral_potential_mapping_Case_studies_from_the_Southern_New_England_Orogen_Australia
#### Tasmania
- https://www.researchgate.net/publication/262380025_Mapping_geology_and_volcanic-hosted_massive_sulfide_alteration_in_the_Hellyer-Mt_Charter_region_Tasmania_using_Random_Forests_TM_and_Self-Organising_Maps
#### Victoria
- https://www.researchgate.net/publication/323856713_Lithological_mapping_using_Random_Forests_applied_to_geophysical_and_remote_sensing_data_a_demonstration_study_from_the_Eastern_Goldfields_of_Australia
- https://publications.csiro.au/publications/#publication/PIcsiro:EP123339/SQmineral%20prospectivity/RP1/RS50/RORECENT/STsearch-by-keyword/LISEA/RI16/RT26 [nickel]
- https://www.researchgate.net/publication/257026553_Regional_prospectivity_analysis_for_hydrothermal-remobilised_nickel_mineral_systems_in_western_Victoria_Australia
### Western Australia
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

## Endowment Modelling
- https://www.researchgate.net/publication/248211962_A_new_method_for_spatial_centrographic_analysis_of_mineral_deposit_clusters
- https://www.researchgate.net/publication/275620329_A_Time-Series_Audit_of_Zipf's_Law_as_a_Measure_of_Terrane_Endowment_and_Maturity_in_Mineral_Exploration
- https://www.researchgate.net/publication/341087909_Assessing_the_variability_of_expert_estimates_in_the_USGS_Three-part_Mineral_Resource_Assessment_Methodology_A_call_for_increased_skill_diversity_and_scenario-based_training
- https://github.com/iagoslc/ZipfsLaw_Quadrilatero_Ferrifero
- https://www.researchgate.net/publication/222834436_Controls_on_mineral_deposit_occurrence_inferred_from_analysis_of_their_spatial_pattern_and_spatial_association_with_geological_features
- https://www.researchgate.net/publication/229792860_From_Predictive_Mapping_of_Mineral_Prospectivity_to_Quantitative_Estimation_of_Number_of_Undiscovered_Prospects
- https://www.researchgate.net/publication/330994502_Global_Grade-and-Tonnage_Modeling_of_Uranium_deposits
- https://pubs.geoscienceworld.org/segweb/economicgeology/article-abstract/103/4/829/127993/Linking-Mineral-Deposit-Models-to-Quantitative?redirectedFrom=fulltext
- https://www.researchgate.net/publication/238365283_Metal_endowment_of_cratons_terranes_and_districts_Insights_from_a_quantitative_analysis_of_regions_with_giant_and_super-giant_deposits
- https://www.researchgate.net/publication/308778798_Spatial_analysis_of_mineral_deposit_distribution_A_review_of_methods_and_implications_for_structural_controls_on_iron_oxide-copper-gold_mineralization_in_Carajas_Brazil
- https://www.researchgate.net/publication/229347041_Predictive_mapping_of_prospectivity_and_quantitative_estimation_of_undiscovered_VMS_deposits_in_Skellefte_district_Sweden
- https://www.researchgate.net/publication/342405763_Predicting_grade-tonnage_characteristics_of_undiscovered_mineralisation_application_of_the_USGS_Three-part_Undiscovered_Mineral_Resource_Assessment_to_the_Sandstone_Greenstone_Belt_of_the_Yilgarn_Bloc
- https://www.sciencedirect.com/science/article/pii/S0169136810000685
- https://www.researchgate.net/publication/240301743_Spatial_statistical_analysis_of_the_distribution_of_komatiite-hosted_nickel_sulfide_deposits_in_the_Kalgoorlie_terrane_Western_Australia_Clustered_or_Not
## World Models
- https://www.researchgate.net/publication/331283650_Archean_crust_and_metallogenic_zones_in_the_Amazonian_Craton_sensed_by_satellite_gravity_data
- https://eartharxiv.org/2kjvc/ -> Global distribution of sediment-hosted metals controlled by craton edge stability
- https://www.researchgate.net/post/Is_it_possible_to_derive_free_air_anomaly_or_bouguer_anomaly_from_gravity_disturbance_data
- https://www.researchgate.net/publication/325344128_The_role_of_basement_control_in_Iron_Oxide-Copper-Gold_mineral_systems_revealed_by_satellite_gravity_models
- https://www.researchgate.net/publication/331428028_Supplementary_Material_for_the_paper_Archean_crust_and_metallogenic_zones_in_the_Amazonian_Craton_sensed_by_satellite_gravity_data
- https://www.leouieda.com/pdf/use-the-disturbance.pdf
- https://www.leouieda.com/papers/use-the-disturbance.html
## Financial Forecasting
- https://www.researchgate.net/publication/317137060_Forecasting_copper_prices_by_decision_tree_learning
- https://www.researchgate.net/publication/4874824_Mine_Size_and_the_Structure_of_Costs
## Agent based Modelling
- https://mpra.ub.uni-muenchen.de/62159/ -> Mineral exploration as a game of chance [Agent Based Modelling]

## Spectral Unmixing
- Overviews and examples, with some focus on neural network approaches.
### Neural Networks
- https://www.researchgate.net/publication/388546200_A_Joint_Multi-Scale_Graph_Attention_and_Classify-Driven_Autoencoder_Framework_for_Hyperspectral_Unmixing
- https://www.researchgate.net/publication/224180646_A_neural_network_approach_for_pixel_unmixing_in_hyperspectral_data
- https://www.researchgate.net/publication/340690859_A_Supervised_Nonlinear_Spectral_Unmixing_Method_by_Means_of_Neural_Networks
- https://www.researchgate.net/publication/326205017_Classification_of_Hyperspectral_Data_Using_a_Multi-Channel_Convolutional_Neural_Network
- https://www.researchgate.net/publication/339062151_Classification_of_small-scale_hyperspectral_images_with_multi-source_deep_transfer_learning
- https://www.researchgate.net/publication/331824337_Comparative_Analysis_of_Unmixing_Algorithms_Using_Synthetic_Hyperspectral_Data
- https://www.researchgate.net/publication/335501086_Convolutional_Autoencoder_For_Spatial-Spectral_Hyperspectral_Unmixing
- https://www.researchgate.net/publication/341501560_Convolutional_Autoencoder_for_Spectral-Spatial_Hyperspectral_Unmixing
- https://www.researchgate.net/publication/333906204_Deep_convolutional_neural_networks_for_land-cover_classification_with_Sentinel-2_images
- https://www.researchgate.net/publication/356711693_Deep-learning-based_latent_space_encoding_for_spectral_unmixing_of_geological_materials
- https://www.researchgate.net/publication/331505001_Deep_learning_and_its_application_in_geochemical_mapping
- https://www.researchgate.net/publication/332696102_Deep_Learning_for_Classification_of_Hyperspectral_Data_A_Comparative_Review
- https://www.researchgate.net/publication/336889271_Deep_Learning_for_Hyperspectral_Image_Classification_An_Overview
- https://www.researchgate.net/publication/327995228_Deep_Spectral_Convolution_Network_for_Hyperspectral_Unmixing
- https://ieeexplore.ieee.org/abstract/document/10580951 -> Exploring Hybrid Contrastive Learning and Scene-to-Label Information for Multilabel Remote Sensing Image Classification [UNSEEN]
- https://www.researchgate.net/publication/356393038_Generalized_Unsupervised_Clustering_of_Hyperspectral_Images_of_Geological_Targets_in_the_Near_Infrared 
- https://ieeexplore.ieee.org/abstract/document/10588073 -> Hyperspectral Image Classification Using Spatial and Spectral Features Based on Deep Learning [UNSEEN]
- https://www.researchgate.net/publication/333301728_Hyperspectral_Image_Classification_Method_Based_on_CNN_Architecture_Embedding_With_Hashing_Semantic_Feature
- https://www.researchgate.net/publication/323950012_Hyperspectral_Unmixing_Using_A_Neural_Network_Autoencoder
- https://www.researchgate.net/publication/339657313_Hyperspectral_unmixing_using_deep_convolutional_autoencoder
- https://www.researchgate.net/publication/339066136_Hyperspectral_Unmixing_Using_Deep_Convolutional_Autoencoders_in_a_Supervised_Scenario
- https://www.researchgate.net/publication/335878933_LITHOLOGICAL_CLASSIFICATION_USING_MULTI-SENSOR_DATA_AND_CONVOLUTIONAL_NEURAL_NETWORKS
- https://ieeexplore.ieee.org/abstract/document/10551851 -> MSNet: Self-Supervised Multiscale Network With Enhanced Separation Training for Hyperspectral Anomaly Detection
- https://www.researchgate.net/publication/331794887_Nonlinear_Unmixing_of_Hyperspectral_Data_via_Deep_Autoencoder_Networks
- https://ieeexplore.ieee.org/abstract/document/10534107 -> ReSC-net: Hyperspectral Image Classification Based on Attention-Enhanced Residual Module and Spatial-Channel Attention
- https://www.researchgate.net/publication/340961027_Recent_Advances_in_Hyperspectral_Unmixing_Using_Sparse_Techniques_and_Deep_Learning
- https://www.researchgate.net/publication/330272600_Semisupervised_Stacked_Autoencoder_With_Cotraining_for_Hyperspectral_Image_Classification
- https://www.researchgate.net/publication/336097421_Spatial-Spectral_Hyperspectral_Unmixing_Using_Multitask_Learning
- https://www.researchgate.net/publication/312355586_Spectral-Spatial_Classification_of_Hyperspectral_Imagery_with_3D_Convolutional_Neural_Network
- https://meetingorganizer.copernicus.org/EGU2020/EGU2020-10719.html -> Sentinel-2 as a tool for mapping iron-bearing alteration minerals: a case study from the Iberian Pyrite Belt (Southern Spain)
- https://www.researchgate.net/publication/334058881_SSDC-DenseNet_A_Cost-Effective_End-to-End_Spectral-Spatial_Dual-Channel_Dense_Network_for_Hyperspectral_Image_Classification
- https://www.researchgate.net/publication/333497470_Integration_of_auto-encoder_network_with_density-based_spatial_clustering_for_geochemical_anomaly_detection_for_mineral_exploration
- https://www.sciencedirect.com/science/article/pii/S0009281924000473 -> Geochemical characteristics and mapping of Reşadiye (Tokat-Türkiye) bentonite deposits using machine learning and sub-pixel mixture algorithms

### General
- https://www.sciencedirect.com/science/article/pii/S0273117724004861?dgcid=rss_sd_all -> Optimization of machine learning algorithms for remote alteration mapping
- https://www.researchgate.net/publication/337841253_A_solar_optical_hyperspectral_library_of_rare_earth-bearing_minerals_rare_earth_oxides_copper-bearing_minerals_and_Apliki_mine_surface_samples
- https://ieeexplore.ieee.org/document/10536904 -> A Reversible Generative Network for Hyperspectral Unmixing With Spectral Variability
- https://www.researchgate.net/publication/3204295_Abundance_Estimation_of_Spectrally_Similar_Minerals_by_Using_Derivative_Spectra_in_Simulated_Annealing
- https://www.researchgate.net/publication/338371376_Accuracy_assessment_of_hydrothermal_mineral_maps_derived_from_ASTER_images
- https://www.researchgate.net/publication/337790490_Analysis_of_Most_Significant_Bands_and_Band_Ratios_for_Discrimination_of_Hydrothermal_Alteration_Minerals
- https://www.researchgate.net/project/Deep-Learning-for-Remote-Sensing-2
- https://ieeexplore.ieee.org/abstract/document/10589462 -> Deep Spectral Spatial Feature Enhancement through Transformer for Hyperspectral Image Classification
- https://www.researchgate.net/publication/331876006_Fusion_of_Landsat_and_Worldview_Images
- https://www.researchgate.net/publication/259096595_Geological_mapping_using_remote_sensing_data_A_comparison_of_five_machine_learning_algorithms_their_response_to_variations_in_the_spatial_distribution_of_training_data_and_the_use_of_explicit_spatial_
- https://www.researchgate.net/publication/341802637_Improved_k-means_and_spectral_matching_for_hyperspectral_mineral_mapping
- https://www.researchgate.net/publication/272565561_Integration_and_Analysis_of_ASTER_and_IKONOS_Images_for_the_Identification_of_Hydrothermally-_Altered_Mineral_Exploration_Sites
- https://www.researchgate.net/publication/236271149_Multi-_and_hyperspectral_geologic_remote_sensing_A_review_GRSG_Member_News
- https://www.researchgate.net/publication/220492175_Multi-and_Hyperspectral_geologic_remote_sensing_A_review
- https://www.sciencedirect.com/science/article/pii/S1574954124001572 -> Rapid estimation of soil Mn content by machine learning and soil spectra in large-scale
- https://www.researchgate.net/publication/342184377_remotesensing-12-01239-v2_1
- https://www.researchgate.net/project/Remote-sensing-exploration-of-critical-mineral-deposits
- https://www.researchgate.net/project/Sentinel-2-MSI-for-geological-remote-sensing
- https://www.researchgate.net/publication/323808118_Thermal_infrared_multispectral_remote_sensing_of_lithology_and_mineralogy_based_on_spectral_properties_of_materials
- https://www.researchgate.net/publication/340505978_Unsupervised_and_Supervised_Feature_Extraction_Methods_for_Hyperspectral_Images_Based_on_Mixtures_of_Factor_Analyzers
### Africa
- https://www.researchgate.net/publication/235443308_Application_of_remote_sensing_and_GIS_mapping_to_Quaternary_to_recent_surficial_sediments_of_the_Central_Uranium_district_Namibia
- https://www.researchgate.net/publication/342373512_Geological_mapping_using_Random_Forests_applied_to_Remote_Sensing_data_a_demonstration_study_from_Msaidira-Souk_Al_Had_Sidi_Ifni_inlier_Western_Anti-Atlas_Morocco
- https://www.researchgate.net/publication/340534611_Identifying_high_potential_zones_of_gold_mineralization_in_a_sub-tropical_region_using_Landsat-8_and_ASTER_remote_sensing_data_a_case_study_of_the_Ngoura-Colomines_goldfield_Eastern_Cameroon
- https://www.researchgate.net/publication/342162988_Lithological_and_alteration_mineral_mapping_for_alluvial_gold_exploration_in_the_south_east_of_Birao_area_Central_African_Republic_using_Landsat-8_Operational_Land_Imager_OLI_data
- https://www.researchgate.net/publication/329193841_Mapping_Copper_Mineralisation_using_EO-1_Hyperion_Data_Fusion_with_Landsat_8_OLI_and_Sentinel-2A_in_Moroccan_Anti_Atlas
- https://www.researchgate.net/publication/230918249_SPECTRAL_REMOTE_SENSING_OF_HYDROTHERMAL_ALTERATION_ASSOCIATED_WITH_VOLCANOGENIC_MASSIVE_SULPHIDE_DEPOSITS_GOROB-HOPE_AREA_NAMIBIA
- https://www.researchgate.net/publication/337304180_The_application_of_day_and_night_time_ASTER_satellite_imagery_for_geothermal_and_mineral_mapping_in_East_Africa
- https://www.researchgate.net/publication/336823002_Towards_Multiscale_and_Multisource_Remote_Sensing_Mineral_Exploration_Using_RPAS_A_Case_Study_in_the_Lofdal_Carbonatite-Hosted_REE_Deposit_Namibia
- https://www.researchgate.net/publication/338296843_Use_of_the_Sentinel-2A_Multispectral_Image_for_Litho-Structural_and_Alteration_Mapping_in_Al_Glo'a_Map_Sheet_150000_Bou_Azzer-El_Graara_Inlier_Central_Anti-Atlas_Morocco
### Brazil
- https://www.researchgate.net/publication/287950835_Altimetric_and_aeromagnetometric_data_fusion_as_a_tool_of_geological_interpretation_the_example_of_the_Carajas_Mineral_Province_PA
- https://www.researchgate.net/publication/237222985_Analise_e_integracao_de_dados_do_SAR-R99B_com_dados_de_sensoriamento_remoto_optico_e_dados_aerogeofisicos_na_regiao_dos_depositos_de_oxido_de_Fe-Cu-Au_tipo_Sossego_e_118_na_Provincia_Mineral_de_Caraja
- https://www.researchgate.net/publication/327503453_Comparison_of_Altered_Mineral_Information_Extracted_from_ETM_ASTER_and_Hyperion_data_in_Aguas_Claras_Iron_Ore_Brazil
- https://www.researchgate.net/publication/251743903_Enhancement_Of_Landsat_Thematic_Mapper_Imagery_For_Mineral_Prospecting_In_Weathered_And_Vegetated_Terrain_In_SE_Brazil
- https://www.researchgate.net/publication/228854234_Hyperspectral_Data_Processing_For_Mineral_Mapping_Using_AVIRIS_1995_Data_in_Alto_Paraiso_de_Goias_Central_Brazil
- https://www.researchgate.net/publication/326612136_Mapping_Mining_Areas_in_the_Brazilian_Amazon_Using_MSISentinel-2_Imagery_2017
- https://www.researchgate.net/publication/242188704_MINERALOGICAL_CHARACTERIZATION_AND_MAPPING_USING_REFLECTANCE_SPECTROSCOPY_AN_EXPERIMENT_AT_ALTO_DO_GIZ_PEGMATITE_IN_THE_SOUTH_PORTION_OF_BORBOREMA_PEGMATITE_PROVINCE_BPP_NORTHEASTERN_BRAZIL
### China
- https://www.researchgate.net/publication/338355143_A_comprehensive_scheme_for_lithological_mapping_using_Sentinel-2A_and_ASTER_GDEM_in_weathered_and_vegetated_coastal_zone_Southern_China
- https://www.researchgate.net/publication/332957713_Data_mining_of_the_best_spectral_indices_for_geochemical_anomalies_of_copper_A_study_in_the_northwestern_Junggar_region_Xinjiang
- https://www.researchgate.net/publication/380287318_Machine_learning_model_for_deep_exploration_Utilizing_short_wavelength_infrared_SWIR_of_hydrothermal_alteration_minerals_in_the_Qianchen_gold_deposit_Jiaodong_Peninsula_Eastern_China
- https://www.researchgate.net/publication/304906898_Remote_sensing_and_GIS_prospectivity_mapping_for_magmatic-hydrothermal_base-_and_precious-metal_deposits_in_the_Honghai_district_China
### Greenland
- https://www.researchgate.net/publication/326655551_Application_of_Multi-Sensor_Satellite_Data_for_Exploration_of_Zn-Pb_Sulfide_Mineralization_in_the_Franklinian_Basin_North_Greenland
- https://www.researchgate.net/publication/337512735_Fusion_of_DPCA_and_ICA_algorithms_for_mineral_detection_using_Landsat-8_spectral_bands
- https://www.researchgate.net/publication/336684298_Landsat-8_Advanced_Spaceborne_Thermal_Emission_and_Reflection_Radiometer_and_WorldView-3_Multispectral_Satellite_Imagery_for_Prospecting_Copper-Gold_Mineralization_in_the_Northeastern_Inglefield_Mobil
### India
- https://www.researchgate.net/publication/337649256_Automated_lithological_mapping_by_integrating_spectral_enhancement_techniques_and_machine_learning_algorithms_using_AVIRIS-NG_hyperspectral_data_in_Gold-bearing_granite-greenstone_rocks_in_Hutti_India
- https://www.researchgate.net/publication/333816841_Integrated_application_of_AVIRIS-NG_and_Sentinel-2A_dataset_in_altered_mineral_abundance_mapping_A_case_study_from_Jahazpur_area_Rajasthan
- https://www.researchgate.net/publication/339631389_Identification_and_characterization_of_hydrothermally_altered_minerals_using_surface_and_space-based_reflectance_spectroscopy_in_parts_of_south-eastern_Rajasthan_India
- https://www.researchgate.net/publication/338116272_Potential_Use_of_ASTER_Derived_Emissivity_Thermal_Inertia_and_Albedo_Image_for_Discriminating_Different_Rock_Types_of_Aravalli_Group_of_Rocks_Rajasthan
### Iran
- https://www.researchgate.net/publication/338336181_A_Remote_Sensing-Based_Application_of_Bayesian_Networks_for_Epithermal_Gold_Potential_Mapping_in_Ahar-Arasbaran_Area_NW_Iran
- https://www.researchgate.net/publication/338371376_Accuracy_assessment_of_hydrothermal_mineral_maps_derived_from_ASTER_images
- https://www.researchgate.net/publication/340606566_Application_of_Landsat-8_Sentinel-2_ASTER_and_WorldView-3_Spectral_Imagery_for_Exploration_of_Carbonate-Hosted_Pb-Zn_Deposits_in_the_Central_Iranian_Terrane_CIT
- https://www.researchgate.net/publication/331428927_Comparison_of_Different_Algorithms_to_Map_Hydrothermal_Alteration_Zones_Using_ASTER_Remote_Sensing_Data_for_Polymetallic_Vein-Type_Ore_Exploration_Toroud-Chahshirin_Magmatic_Belt_TCMB_North_Iran
- https://www.researchgate.net/publication/327832371_Band_Ratios_Matrix_Transformation_BRMT_A_Sedimentary_Lithology_Mapping_Approach_Using_ASTER_Satellite_Sensor
- https://www.researchgate.net/publication/331314687_Lithological_mapping_in_Sangan_region_in_Northeast_Iran_using_ASTER_satellite_data_and_image_processing_methods
- https://www.researchgate.net/publication/330774780_Mapping_hydrothermal_alteration_zones_and_lineaments_associated_with_orogenic_gold_mineralization_using_ASTER_data_A_case_study_from_the_Sanandaj-Sirjan_Zone_Iran
- https://www.researchgate.net/publication/380812370_Optimization_of_machine_learning_algorithms_for_remote_alteration_mapping
- https://www.researchgate.net/publication/362620968_Spatial_mapping_of_hydrothermal_alterations_and_structural_features_for_gold_and_cassiterite_exploration 
### Peru
- https://www.researchgate.net/publication/271714561_Geology_and_Hydrothermal_Alteration_of_the_Chapi_Chiara_Prospect_and_Nearby_Targets_Southern_Peru_Using_ASTER_Data_and_Reflectance_Spectroscopy
- https://www.researchgate.net/publication/317141295_Hyperspectral_remote_sensing_applied_to_mineral_exploration_in_southern_Peru_A_multiple_data_integration_approach_in_the_Chapi_Chiara_gold_prospect
### Spain
- https://www.researchgate.net/publication/233039694_Geological_mapping_using_Landsat_Thematic_Mapper_imagery_in_Almeria_Province_south-east_Spain
- https://www.researchgate.net/publication/263542786_WEIGHTS_DERIVED_FROM_HYPERSPECTRAL_DATA_TO_FACILITATE_AN_OPTIMAL_FIELD_SAMPLING_SCHEME_FOR_POTENTIAL_MINERALS
### Other
- https://www.researchgate.net/publication/341611032_ASTER_spectral_band_ratios_for_lithological_mapping_A_case_study_for_measuring_geological_offset_along_the_Erkenek_Segment_of_the_East_Anatolian_Fault_Zone_Turkey
- https://www.researchgate.net/publication/379960654_From_sensor_fusion_to_knowledge_distillation_in_collaborative_LIBS_and_hyperspectral_imaging_for_mineral_identification
- https://www.researchgate.net/publication/229383008_Hydrothermal_Alteration_Mapping_at_Bodie_California_using_AVIRIS_Hyperspectral_Data
- https://www.researchgate.net/publication/332737573_Identification_of_alteration_zones_using_a_Landsat_8_image_of_densely_vegetated_areas_of_the_Wayang_Windu_Geothermal_field_West_Java_Indonesia
- https://www.researchgate.net/publication/325137721_Interpretation_of_surface_geochemical_data_and_integration_with_geological_maps_and_Landsat-TM_images_for_mineral_exploration_from_a_portion_of_the_precambrian_of_Uruguay
- https://www.researchgate.net/publication/336684298_Landsat-8_Advanced_Spaceborne_Thermal_Emission_and_Reflection_Radiometer_and_WorldView-3_Multispectral_Satellite_Imagery_for_Prospecting_Copper-Gold_Mineralization_in_the_Northeastern_Inglefield_Mobil
- https://www.researchgate.net/publication/304036250_Mineral_Exploration_for_Epithermal_Gold_in_Northern_Patagonia_Argentina_From_Regional-_to_Deposit-Scale_Prospecting_Using_Landsat_TM_and_Terra_ASTER
- https://www.researchgate.net/publication/340652300_New_logical_operator_algorithms_for_mapping_of_hydrothermally_altered_rocks_using_ASTER_data_A_case_study_from_central_Turkey
- https://www.researchgate.net/publication/324938267_Regional_geology_mapping_using_satellite-based_remote_sensing_approach_in_Northern_Victoria_Land_Antarctica
- https://www.mdpi.com/2072-4292/17/11/1878 -> Scalable Hyperspectral Enhancement via Patch-Wise Sparse Residual Learning: Insights from Super-Resolved EnMAP Data

### NLP
- https://www.researchgate.net/publication/390874026_Assessing_named_entity_recognition_by_using_geoscience_domain_schemas_the_case_of_mineral_systems
- https://link.springer.com/article/10.1007/s12371-024-01011-2 -> Can AI Get a Degree in Geoscience? Performance Analysis of a GPT-Based Artificial Intelligence System Trained for Earth Science (GeologyOracle)
- https://www.researchgate.net/publication/376671309_Enhancing_knowledge_discovery_from_unstructured_data_using_a_deep_learning_approach_to_support_subsurface_modeling_predictions
- https://www.mdpi.com/2220-9964/13/7/260 -> Extracting Geoscientific Dataset Names from the Literature Based on the Hierarchical Temporal Memory Model
- Knowledge-Infused LLM Application in Data Analytics: Using Mindat as an Example
- https://www.sciencedirect.com/science/article/pii/S0169136824002154 -> Three-dimensional mineral prospectivity mapping based on natural language processing and random forests: A case study of the Xiyu diamond deposit, China

### LLM
- https://arxiv.org/pdf/2401.16822 - EarthGPT: A Universal Multi-modal Large Language Model for Multi-sensor Image Comprehension in Remote Sensing Domain
- https://link.springer.com/article/10.1007/s12371-024-01011-2 -> Can AI Get a Degree in Geoscience? Performance Analysis of a GPT-Based Artificial Intelligence System Trained for Earth Science (GeologyOracle)
  - Geology Oracle web prototype - https://geologyoracle.com/ask-the-geologyoracle/
  - Knowledge-Infused LLM Application in Data Analytics: Using Mindat as an Example -> https://www.proquest.com/openview/38854958cb460de71484a93584fe0ff4/1?cbl=18750&diss=y&pq-origsite=gscholar [UNSEEN PAST THIS]
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5242692 -> GeoProspect: a domain specific geological large language model

# General-Interest
- https://arxiv.org/abs/2404.05746v1 -> Causality for Earth Science -> A Review on Time-series and Spatiotemporal Causality Methods
- https://link.springer.com/article/10.1007/s11831-025-10244-5 -> Deep Learning for Time Series Forecasting: Review and Applications in Geotechnics and Geosciences
- https://ieeexplore.ieee.org/abstract/document/10825956 -> Enabling Scalable Mineral Exploration:Self-Supervision and Explainability
- https://pure.au.dk/ws/portalfiles/portal/429062897/Fast_Correct_Clustering_in_Time_and_Space_using_the_GPU-Katrine_Scheel_Killmann.pdf -> Fast Correct Clustering in Time and Space using the GPU
- https://www.researchgate.net/publication/384137154_Guidelines_for_Sensitivity_Analyses_in_Process_Simulations_for_Solid_Earth_Geosciences
- https://www.mdpi.com/1660-4601/18/18/9752 -> Learning and Expertise in Mineral Exploration Decision-Making: An Ecological Dynamics Perspective
- https://www.sciencedirect.com/science/article/pii/S2214629624001476 -> Mapping critical minerals projects and their intersection with Indigenous peoples' land rights in Australia
- https://www.sciencedirect.com/science/article/pii/S0169136824003470 -> Overcoming survival bias in targeting mineral deposits of the future: Towards null and negative tests of the exploration search space, accounting for lack of visibility
- https://www.sciencedirect.com/science/article/pii/S088329272400115X -> Ranking Mineral Exploration Targets in Support of Commercial Decision Making: A Key Component for Inclusion in an Exploration Information System
- https://earthmover.io/blog/tensors-vs-tables -> Tensors vs tables - why array native data structures are needed for performance


## Deep Learning
- https://www.researchgate.net/publication/390346411_A_Biological-Inspired_Deep_Learning_Framework_for_Big_Data_Mining_and_Automatic_Classification_in_Geosciences
- https://arxiv.org/abs/2408.11804 -> Approaching Deep Learning through the Spectral Dynamics of Weights
- https://www.researchgate.net/publication/384833877_A_Review_of_Mineral_Prospectivity_Mapping_Using_Deep_Learning 
- https://arxiv.org/pdf/2310.19909.pdf -> Battle of the Backbones: A Large-Scale Comparison of Pretrained Models across Computer Vision Tasks
- https://pure.mpg.de/rest/items/item_3029184_8/component/file_3282959/content -> Deep learning and process understanding for data-driven Earth system science
- https://www.tandfonline.com/doi/pdf/10.1080/17538947.2024.2391952 -> Deep learning for spatiotemporal forecasting in Earth system science: a review
- https://wires.onlinelibrary.wiley.com/doi/pdfdirect/10.1002/widm.1554 -> From 3D point-cloud data to explainable geometric deep learning: State-of-the-art and future challenges
- https://arxiv.org/pdf/2410.16602 -> Foundation Models for Remote Sensing and Earth Observation: A Survey
- https://www.researchgate.net/publication/383460665_Poly2Vec_Polymorphic_Encoding_of_Geospatial_Objects_for_Spatial_Reasoning_with_Deep_Neural_Networks
- https://www.nature.com/articles/s41467-021-24025-8 -> Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data
-https://arxiv.org/html/2401.10825v3 -> Recent Advances in Named Entity Recognition: A Comprehensive Survey and Comparative Study
- https://arxiv.org/abs/2404.07738 ResearchAgent: Iterative Research Idea Generation over Scientific Literature with Large Language Models
- https://ieeexplore.ieee.org/abstract/document/10605826 -> Swin-CDSA: The Semantic Segmentation of Remote Sensing Images Based on Cascaded Depthwise Convolution and Spatial Attention Mechanism
- https://www.sciencedirect.com/science/article/abs/pii/S0098300424000839#sec6 -> Leveraging automated deep learning (AutoDL) in geosciences

## Data Types
- Paul Bourke GOCAD formats https://paulbourke.net/dataformats/gocad/gocad.pdf
