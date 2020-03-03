import spacy
activated = spacy.prefer_gpu()
import json
import os
import re
from flask import render_template,Flask,request,url_for
from textblob import TextBlob 
from flaskext.markdown import Markdown
from spacy import displacy

import sys
# print(get_models(MODEL_PATH), file=sys.stderr)

app = Flask(__name__,static_url_path="",static_folder="img")
Markdown(app)

# HTML_WRAPPER for spaCy NER rendering.
INPUT_WRAPPER = """<div style="overflow-x:auto;padding:1rem"><div class="entities" style="line-height:2.5">{}</div></div>"""
OUTPUT_WRAPPER = """<div style="overflow-x:auto;padding:1rem">{}</div>"""

# Path which contained the model
MODEL_PATH="./pretrained_model/spacy"

# Preset color and tag template for symptoms, drugs and dosage.
NER_PRESET_COLORS = {"SYMP": "#9370DB", "DRUG": "#F58B4C", "DOSE": "#B5C689"}
OPTIONS = {"ents": ["SYMP","DRUG","DOSE"], "colors": NER_PRESET_COLORS}

def get_models(directory):
	# Input the file directory of the models (hard coded)
	# Return a dictionary of pretrained model name and it's path.
	result = {}

	# Add 2 spaCy pretrained default models.
	#result["en_core_web_sm"]="en_core_web_sm"
	#result["en_core_web_md"]="en_core_web_md"

	for file in os.listdir(directory):
		result[file]=os.path.join(directory, file)
	return(result)

def clean_string(string):
	# Input a string representation
	# Return processed string with no twitter hash tag.
	return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", string).split())
  
def get_polarity(sentence):
	# Sentimental Analysis function, using pretrained model
	# TextBlob - Trained on Twitter dataset
	# Return a polarity score representing sentiment
	try:
		analysis = TextBlob(sentence)
		return(analysis.sentiment.polarity)
	except:
		return(0.5)

def get_sentiment(polarity):
	# Input polarity value
	# Return sentiment
	try:
		if polarity > 0: 
			return('Positive')
		elif polarity == 0: 
			return('Neutral')
		else: 
			return('Negative')
	except:
		return('Neutral')

@app.route('/',methods=["GET"])
def index_page():
	# Landing page of the application.
	# Require model paths to be load.
	model_name = get_models(MODEL_PATH)
	return render_template('index.html', model_name=model_name)

@app.route('/results',methods=["GET","POST"])
def result_page():
	if request.method == 'POST':
		#ent_arr = request.form['render_drug'] + request.form['render_symptom'] + request.form['render_dose']
		OPTIONS["ents"] = request.form.getlist('render_ner_list')

		# Initiate model depending on user's choice.
		selected_model = request.form['selected_model']
		nlp = spacy.load(os.path.join(selected_model))

		# Extract and analyse text
		raw_text = request.form['input_text']
		docx = nlp(raw_text)

		# Formatting input text.
		input_text = raw_text.replace("\n","</br>")
		input_text = INPUT_WRAPPER.format(input_text)

		# Disable color scheme for spaCy model.
		if selected_model == "en_core_web_sm" or selected_model == "en_core_web_md":
			html = displacy.render(docx,style="ent")
		else:
			html = displacy.render(docx,style="ent",options=OPTIONS)

		html = html.replace("\n\n","\n")
		output_text = OUTPUT_WRAPPER.format(html)

		# Initialise the drop-box again.
		model_name = get_models(MODEL_PATH)
		
		# Polarity Analysis
		raw_text = clean_string(raw_text)
		polarity = get_polarity(raw_text)
		text_polarity = ((polarity + 1.0) / 2.0) * 100.0
		other_polarity = 100.0 - text_polarity

		# Sentiment analysis
		text_sentiment = get_sentiment(polarity)

		return render_template('result.html', input_text=input_text, output_text=output_text, model_name=model_name, text_polarity=text_polarity, other_polarity=other_polarity, text_sentiment=text_sentiment)

# About page of the application.
# Self-introduction, motivation, summarise.
@app.route('/about')
def about_me():
	return render_template('about.html')

# Flask debug function.
if __name__ == '__main__':
	app.run(debug=True)