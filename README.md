###### Please Follow These Steps to Make It Work
Since the paper presented the same results on all 3 sites (Rome, Florence & Istanbul) and presented results only on Rome, we did the same.
Important: Don't proceed to the next step without finishing the current one.

Preparations:
1) Manually download GitHub data from Flickr (Italy): "https://github.com/igobrilhante/TripBuilder" Extract the zip file and place it as-is in the "data" directory following the scheme below.

2) In PowerShell create a new environment "python -m venv recsys_env"
3) Activate the environment "recsys_env\Scripts\activate"
4) Install all requirements "pip install -r requirements.txt" This might cause problems, open the requirements file and install in batches by commenting out some rows and installing them afterward

Simulation:
5) In PowerShell verify the environment is activated "recsys_env\Scripts\activate"
6) In PowerShell run "python src/exp/proc_Flickr.py"

7) In GitBash activate the environment "source recsys_env/Scripts/activate" 
8) In GitBash run "./run1_R.sh"
9) In GitBash run "./run2_R.sh"

10) In PowerShell run "python src/exp/runx_batch.py --city Rome" 

All pk files created in out\experiments directory
11) Plot - Manually run the Python file "\src\exp\Analysis_compare_Models.py" to get the original result plot, or use "plot.ipynb" as provided in the original paper.


To plot each improvement, you need to use the relevant files - overwrite files from the improvement folder to "\src\exp".
Improvement 1
1) Copy from "Improvement1_Preferred_Distance_from_CC" and paste into "\src\exp"  
2) Run simulation steps 5-10  
3) Plot - Manually run Python file "\src\exp\Analysis_compare_Models.py" The combined results of the models are relevant - compare these results to the original run.

Improvement 2
1) Copy from "Improvement2_Model_KNN" and paste into "\src\exp"  
2) Run simulation steps 5-10  
3) Plot - Manually run Python files "Analysis_compare_Models_knn.py" and "Analysis_compare_Models_25_50_75.py" to see all results comparisons.

Improvement 3
1) Copy from "Improvement3_Religious_Spiritual" and paste into "\src\exp"  
2) Run simulation steps 5-10  
3) Plot - Manually run Python file "\src\exp\Analysis_compare_Models.py" the Combined results of the models are relevant - need to compare those results to the original run 

Improvement 4
1) Copy from "Improvement4_Reduced_Walking" and paste into "\src\exp"  
2) Run simulation steps 5-10  
3) Copy 2 files "runx_Rome_ease_vae.pk" and "runx_Rome_ease_wmf.pk" from "out/experiments" to "All_pk_comparison" and add suffix "_4_reduce_walking_with_routing" 
4) Copy from "Improvement4_Reduced_Walking_base" and paste into "\src\exp"  
5) Run simulation steps 5-10  
6) Copy 2 files "runx_Rome_ease_vae.pk" and "runx_Rome_ease_wmf.pk" from "out/experiments" to "All_pk_comparison" and add suffix "_4_reduce_walking_base" 
7) Plot - Manually run Python file "\src\exp\Analysis_comp_impro4_reduce_walking_smart_walking.py"


############ To return to the original code ############
1) Copy from "Original_Code" and paste into "\src\exp"  
2) Run simulation steps 5-10 



############################### The original ReadMe File ######################################
"
### Positive-Sum Impact of Multistakeholder Recommender Systems for Urban Tourism Promotion and User Utility

This repository contains source code for replicating experiments for the 
RecSys 2024 conference. It also includes training logs, noting that
hyperparameter tuning and simulations are time-consuming and require 24GB of RAM.

##### In case you plan to replicate experiments

Download this repository and original datasets:
- <https://github.com/igobrilhante/TripBuilder> (full)
- <https://sites.google.com/site/yangdingqi/home/foursquare-dataset> (global-scale check-in dataset)

Prepare `recsys` directory structure and unpack data:
```
(recsys)
|-- data
|   |-- Foursquare33M
|   |   |-- dataset_TIST2015_Checkins.txt
|   |   |-- dataset_TIST2015_Cities.txt
|   |   |-- dataset_TIST2015_POIs.txt
|   |   `-- dataset_TIST2015_readme.txt
|   `-- tripbuilder-dataset-dist
|       |-- assets
|       |   |-- ...
|       |   `-- license.txt
|       |-- florence
|       |   |-- florence-photos.txt
|       |   |-- florence-pois-clusters.txt
|       |   |-- florence-pois.txt
|       |   |-- florence-trajectories.txt
|       |   `-- license.txt
|       |-- index.html
|       |-- license.txt
|       |-- pisa
|       |   |-- license.txt
|       |   |-- pisa-photos.txt
|       |   |-- pisa-pois-clusters.txt
|       |   |-- pisa-pois.txt
|       |   `-- pisa-trajectories.txt
|       `-- rome
|           |-- license.txt
|           |-- rome-photos.txt
|           |-- rome-pois-clusters.txt
|           |-- rome-pois.txt
|           `-- rome-trajectories.txt
|-- log
|-- out
|-- run1_F.sh
|-- run1_I.sh
|-- run1_R.sh
|-- run2_F.sh
|-- run2_I.sh
|-- run2_R.sh
`-- src
    `-- exp
        |-- __init__.py
        |-- collaborative_filtering.py
        |-- datafactory.py
        |-- environment.py
        |-- proc_Flickr.py
        |-- proc_Foursquare.py
        |-- recommender.py
        |-- run1.py
        |-- run2.py
        |-- runx.py
        |-- runx_batch.py
        |-- sim.py
        |-- ubm.py
        `-- utils.py
```

0. Run from `recsys` directory:
```
python src/exp/proc_Flickr.py
<!-- python src/exp/proc_Foursquare.py -->
```
It will process source data, `separate` local residents from tourists, 
and apply Core-filtering.

![Img1](img/appendix_data.png "Data")

1. Run:
```
./run1_R.sh
./run1_F.sh
./run1_I.sh
```
In a sequence, or parallel from different terminal screens.
It will process Rome (R), Florence (F), and Istanbul (I) true user preferences 
and save results (`out/` dir) and corresponding training logs (`log/` dir) to disk. 

2. Run:
```
./run2_R.sh
./run2_F.sh
./run2_I.sh
```
In a sequence, or parallel from different terminal screens.
It will estimate limited awareness set for each user and calibrate 
multinomial choice model, and save results (`out/` dir) and corresponding 
training logs (`log/` dir) to disk.

3. Run:
```
python src/exp/runx_batch.py --city Rome
python src/exp/runx_batch.py --city Florence
python src/exp/runx_batch.py --city Istanbul
```
(sequentially!) because each run is already executed in 
parallel and consumes 24GB of RAM memory. This step will produce both: 
experiment artefacts (`.pk` files in `out/experiments/`) 
and experiment logs (`log/` dir).

4. Run in order to reproduce our Figure:
```
plot.ipynb
```

![Img2](img/Rome_results.png "Results")

requirements.txt:
```
cornac==2.1.0
h5py==3.10.0
numba==0.59.1
numpy==1.26.4
polars==0.20.18
```
"