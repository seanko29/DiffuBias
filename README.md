# DiffuBias
Official Repository for [ACM SAC'25] Debiasing Classifiers by Amplifying Bias with Latent Diffusion and Large Language Models

![Diffubias](https://github.com/user-attachments/assets/5335a13b-09bf-4bf5-b031-99b068b33fda)

## Requirements
```
pip install -r requirements.txt
```

## Dataset: 
Dataset can be found [here](https://drive.google.com/drive/folders/1z6gQKa_Kgoj6ZOJHsIQ-BPMnU3qMImBN?usp=sharing)


## Step 1: Make a biased classifier f_b 
- Train on a specific benchmark dataset (CMNIST, BAR, BFFHQ, Dogs n Cats)

`````
python classifier.py
`````



## Step 2: Extract Bias 
`````
# extract bias-conflict data using top-K loss
python extract_bias.py 
`````

## Step 3: Text-Captioning
```
# text-captioning using BLIP model
python blip.py
```
## Step 3.5: Text-Filtering
```
# stop_word filter
python stop_word_filter.py --input_csv /path/to/your/input.csv --output_csv filtered_results.csv
```

## Step 4: Amplify bias-conflict images 
```
# Example for dataset BAR 1pct, with 10,000 images generation on the filtered text.
python stable_diffusion.py --csv_path path/to/filter_csv_file.csv --dataset_name bar_1pct --max_images 10000
```

## Step 5: Debias Classifier
```
python final_classifier.py
```

[Final weight](https://drive.google.com/drive/folders/1Akzr8LdsM0oyJ6-YjqiGUF9Y2_rAFjL6?usp=sharing)
