from flask import Flask,url_for,render_template,request
import spacy
from spacy import displacy
nlp = spacy.load('C:/Users/khoin/Desktop/project_code/third_year_project/pretrained_model/spacy/temp')
import json

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

from flaskext.markdown import Markdown

#app = Flask(__name__)
app = Flask(__name__,static_url_path="",static_folder="img")
Markdown(app)

#doc = nlp("We describe a patient with a liver abscess due to Entamoeba histolytica, in whom metronidazole therapy (total dose, 21 g over 14 days) was complicated by reversible deafness, tinnitus, and ataxia and who relapsed 5 months later with a splenic abscess.")
#for ent in doc.ents:
#    print(ent.label_, ent.text)
#exit()

# def analyze_text(text):
# 	return nlp(text)

@app.route('/')
def index():
	# raw_text = "Bill Gates is An American Computer Scientist since 1986"
	# docx = nlp(raw_text)
	# html = displacy.render(docx,style="ent")
	# html = html.replace("\n\n","\n")
	# result = HTML_WRAPPER.format(html)

	return render_template('index.html')


@app.route('/extract',methods=["GET","POST"])
def extract():
	if request.method == 'POST':
		raw_text = request.form['rawtext']
		docx = nlp(raw_text)
		# Preset template for symptoms and drugs.
		colors = {"SYMP": "#9370DB", "DRUG": "#F58B4C", "DOSE": "#B5C689"}
		options = {"ents": ["SYMP","DRUG","DOSE"], "colors": colors}

		# NER for Symptom and Drug names.
		#doc=nlp(sample_sentence)
		#displacy.render(doc, style="ent", jupyter=True, options=options)
		html = displacy.render(docx,style="ent", options=options)
		html = html.replace("\n\n","\n")
		result = HTML_WRAPPER.format(html)

	return render_template('result.html',rawtext=raw_text,result=result)


@app.route('/previewer')
def previewer():
	return render_template('previewer.html')

@app.route('/preview',methods=["GET","POST"])
def preview():
	if request.method == 'POST':
		newtext = request.form['newtext']
		result = newtext

	return render_template('preview.html',newtext=newtext,result=result)


if __name__ == '__main__':
	app.run(debug=True)