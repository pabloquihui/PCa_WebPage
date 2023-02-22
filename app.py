# from flask import Flask, render_template, redirect, session
from functools import wraps
import pymongo
import os
from flask import Flask, render_template, flash, redirect, url_for, request, send_from_directory, session
from werkzeug.utils import secure_filename
import re
import numpy as np
from config import Config

app = Flask(__name__)
app.secret_key = b'\xb4\xc6O8\xf3\xba\x83p\xc2\x18-b\xe1\xadqL'
app.config.from_object(Config)

# Database
client = pymongo.MongoClient('localhost', 27017)
db = client.user_login_system

from user.utils import make_png, read_img, save_png
# Decorators
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            return redirect('/')
    return wrap

# Routes
from user import routes

@app.route('/')
def home():
    return render_template('login.html', title='Log In')

@app.route('/index/')
@login_required
def index():
    path = app.config['UPLOAD_FOLDER'] + '/npy'
    files = [file for file in os.listdir(path)
            if (file.endswith('.npy'))]
    return render_template('index.html', title='Home', files=files)

@app.route('/upload/')
@login_required
def upload_file():
    return render_template('upload.html', title='Upload files')

@app.route('/uploader/', methods=['POST'])
def upload_file_():
    cause = ''  # For catching specific cause of error
    pat = re.compile(r'[^\.]*\.(.*)')
    formt = re.findall(pat, request.files['mri'].filename)[0]

    # formats = pat.findall(request.files[m].filename)[0])
    if request.method == 'POST':
        try:
            cause = 'while uploading the files. Ensure that the files'
            ' are accessible and try again. '
            path = app.config['UPLOAD_FOLDER']
            path = f'{path}/orig'
            f = request.files['mri']
            f.save(
                os.path.join(
                path,
                secure_filename(f"t2w_mri.{formt}")
                )
            )
            flash('Files were uploaded succesfully.')

            cause = 'while exporting the files into a single multimodal-MRI'
            ' (.mmri) file. Make sure the uploaded files are valid MRI files'
            ' and try again.'
            img = read_img(os.path.join(path,                                       # TODO: REVISAR AQUI XQ NO JALA AL CARGAR IMAGEN
                secure_filename(f"t2w_mri.{formt}")))
            np.save(os.path.join(
              (app.config['UPLOAD_FOLDER']+'/npy'),
              secure_filename(f"{request.form['name']}")
            ), img)

            cause = None

        except Exception as e:
            flash(
                f'An error occured {cause}' if cause is not None else
                'An unknown error occured.')
            return f"""<div class="w3-container">
              <h1 class="w3-xxxlarge w3-text-black"><b>Sorry Something Went Wrong.</b></h1>
              <hr style="width:50px;border:5px solid red" class="w3-round">
              <p>An error occured while uploading the MRI files. See below for more info.</p>
              <br />
              <h3 class="w3-xlarge w3-text-black"><b>Error Text:</b></h3>
              <hr>
              <p> {e} </p>
              <a href='/upload'><h3 class="w3-xlarge w3-text-black">
                <b>&lt; Go back and try again.</b></h3></a>
            </div>"""
        return redirect(url_for('index'))
    return redirect(url_for('index'))

@app.route('/analyze/', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        file = request.form.get('files_dropdown')
        print(file)
        print(app.config["UPLOAD_FOLDER"])
        # try:
        out_file = make_png(file)
        success = True
        error = None
        # except:
        #     sucess = False
        #     error = 'Error making the png file'


        return render_template('analyze.html', title='Results', success=success,
                               file=out_file, error=error, folder=app.config["UPLOAD_FOLDER"])
    elif request.method == 'GET':
        if app.config['TESTING_ANALYZE']:
            return render_template('analyze.html', title='Testing Analyze', success=True,
                                   file='Test', error='error', folder=app.config["UPLOAD_FOLDER"])
        else:
            flash('Select a MRI file from the list or add your own to get the prediction result.')
            return redirect(url_for('index'))

@app.route('/download-mask/<file>/', methods=['GET'])
def download(file):
    if request.method == 'GET':
        name = '/preds/t2w_pred.png'
        return redirect(url_for(name), code=301)


if __name__ == '__main__':
    app.run()