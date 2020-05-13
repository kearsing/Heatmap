import os
from flask import Flask, flash, request, redirect, url_for, render_template

from werkzeug.utils import secure_filename
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'Temp' #'/home/kearsing/mysite/Temp'
app.config['STATIC_FOLDER'] = 'static' #'/home/kearsing/mysite/templates'
ALLOWED_EXTENSIONS = {'txt', 'csv',}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            global filename
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('upload_file',
            #                         filename=filename))
            global df
            df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename),header=None, names=(['Label', 'Text']), encoding = "ISO-8859-1")
            dfhead = df.head(20)
            # print(dfhead)
            return render_template('simple.html', tables=[dfhead.to_html(classes='data')], titles='First 20 Lines')
    return render_template('upload.html')

@app.route('/VisualMatrix', methods=['GET', 'POST'])
def load():
    return render_template('VisualMatrix.html')
@app.route('/similarity', methods=['GET', 'POST'])
def similarity():

    data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename), header=None, names=(['Label', 'Text']), encoding="ISO-8859-1")
    docs = []
    labels = []
    labelGroup = data.groupby('Label')
    for label in labelGroup.groups.keys():
        # print(cuisine)
        labels.append(label)
        filter = data['Label'] == label
        docs.append(data['Text'].where(filter).dropna().to_string(index=False))

    labels = [i.replace("'", '').replace('&','').replace(' ','_') for i in labels]
    # for doc in docs: count_vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
    count_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 1))
    sparse_matrix = count_vectorizer.fit_transform(docs)

    # print the vocabulary
    word_count = len(count_vectorizer.get_feature_names())

    doc_term_matrix = sparse_matrix.todense()
    df2 = pd.DataFrame(doc_term_matrix,
                      columns=count_vectorizer.get_feature_names(), index=labels)
    # print(cuisines)
    x = pd.DataFrame({n: df2.T[col].nlargest(5).index.tolist()
                      for n, col in enumerate(df2.T)}).T
    # print(x)
    x.index = labels

    # Compute Cosine Similarity
    CS = cosine_similarity(df2, df2)
    # print(CS)
    # pd.options.display.float_format = '{:2f}'.format
    CS_pd = pd.DataFrame(CS, index=labels, columns=labels)
    df = pd.DataFrame(data=CS, index=labels, columns=labels)


    write_file(labels, df)


    return render_template('similarity.html', table1=[CS_pd.round(2).to_html(classes='data')],table2=[x.to_html(classes='data')], title1='Similarity Matrix',
                           title2 = 'Word Popularity', word_count = word_count,labels=','.join(labels))
def write_file(labels,df):

    counter = 0
    outfile = open(os.path.join(app.config['STATIC_FOLDER'],'simValues.csv'), 'w', encoding='UTF-8')
    outfile.writelines('var1,var2,score\n')
    for index, row in df.iterrows():
        for label in labels:
            outfile.writelines(labels[counter] + ',' + label + ',' + str(row[label].round(2)) + '\n')
        counter += 1

if __name__ == '__main__':
   app.run(debug = True)
