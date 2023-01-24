import os
basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = 'secret'
    TEMPLATES_AUTO_RELOAD = True
    UPLOAD_FOLDER = './static/images'
    EXPORT_FOLDER = './static/images/exported'
    # EXPORT_FOLDER_REL = 'images/exported/'
    EXPORT_FOLDER_REL = 'images/'
    MAX_CONTENT_PATH = 5e6
    TESTING_ANALYZE = os.environ.get('TESTING_ANALYZE', 0)
    IMAGE_SIZE = 256
    NO_ZONES = 5
    T = 5                  #Number of predictions for uncertainty quantification

    DEBUG = True
    MODEL = './static/models/mc_dense_unet_model'
