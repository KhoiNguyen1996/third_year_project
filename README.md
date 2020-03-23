# Text Mining Adverse Drug Reactions from Social Media
According to the Food and Drug Administration agency, ADRs are a major public health concern and a leading cause of morbidity and mortality. Currently, information relating to ADRs is fragmented and curated manually from distributed and heterogeneous resources such as social media (online forums). The aim of this project is to extract entities involved in ADRs (diseases, symptoms, drugs, treatments) from social media using text mining tools. In addition, sentiment analysis (positive, negative) can also help to distinguish ADRs as postings indicate negative feelings. 

Related papers:
1. https://www.sciencedirect.com/science/article/pii/S1532046416300508
2. https://academic.oup.com/jamia/article/22/3/671/776531

## Project Objectives
- Input a paragraph or sentence that are drug-related, and the application will be able to detect named entities in the input that are involved in ADRs (symptoms, drugs, or dosage) and will highlight those entities in the input message.
- Furthermore, the application also used a pre-trained Twitter sentimental analysis model in order to analyse the input text and predict whether the input sentence have positive or negative polarity.

## Install required libraries
Open terminal and make sure requirements.txt is in the current directory location. Then execute the command:  
> pip install -r requirements.txt

## Application Demo
Start the **Flask** Server  
> python ner_app.py  

Open a Web Browser e.g Chrome.  
Navigate to the link given in the Terminal.  

# Dataset and NLP model Architecture
## Training Corpus
Development of a Benchmark Corpus to Support the Identification of Adverse Drug Effects from Case Reports  
Harsha Gurulingappa, Abdul Mateen-Rajput, Angus Roberts, Juliane Fluck, Martin Hofmann-Apitius, and Luca Toldo  
Journal of Biomedical Informatics, In Press, 2012.  
The corpus described in the following article will be available for download here.  
Gurulingappa et al., Benchmark Corpus to Support Information Extraction for Adverse Drug Effects, JBI, 2012.  
http://www.sciencedirect.com/science/article/pii/S1532046412000615
https://sites.google.com/site/adecorpus/home

## Named-Entity recognition model
The model I implemented use a deep **Convolutional Neural Network** with *residual connections* for its custom named entity recognition pipeline.  
It's a **statistical model** and it's use a transition-based approach to named entity parsing.

# Results
The best model achieved **83%** F1 score on average accuracy when testing the model using 10-fold cross-validation: **en_core_web_lg**  

Here is the table of the average accuracy of the named-entities recognition model with different word embedding techniques.

| In-app Name    | Model         | Accuracy  |
| -------------  |:-------------:|:---------:|
| en_core_web_lg_sa_2 | Bloom Embedding  |0.8304 ± 0.001|
| bio_word2vec_sa_2 | PubMed Word2Vec Embedding |0.8298 ± 0.001|
| glove          | Glove Embedding       |0.8281 ± 0.001|