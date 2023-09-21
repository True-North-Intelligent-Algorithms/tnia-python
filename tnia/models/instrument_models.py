import json
import os
# import imsave
from skimage.io import imread, imsave
from enum import Enum

class InstrumentModelKeys(Enum):
    MODEL_NAME = "model_name"
    XY_SPACING = "xy_spacing"
    Z_SPACING = "z_spacing"
    XY_SIZE = "xy_size"
    Z_SIZE = "z_size"
    NA = "NA"
    NI = "ni"
    NS = "ns"
    EMISSION_WAVELENGTH = "emission_wavelength"
    MAGNIFICATION = "magnification"
    DEPTH = "depth"
    MODALITY = "modality"
    CONFOCAL_FACTOR = "confocal_factor"
    MODEL_TYPE = "model_type"
    NOTES = "notes"
    FILE = "file"

class InstrumentModalities(Enum):
    WIDEFIELD = "widefield"
    CONFOCAL = "confocal"
    TWO_PHOTON = "two-photon"
    LIGHT_SHEET = "light-sheet"
    SPINNING_DISK = "spinning-disk"

class ModelTypes(Enum):
    THEORETICAL_PSF = "theoretical-psf"
    MEASURED_PSF = "measured-psf"
    STARDIST_NETWORK = "stardist-network"
    CARE_NETWORK = "care-network"
    DEEPLEARNING_NETWORK = "deeplearning-network"

def get_model_file(settings, json_data):
    """Get the file name of the model that matches the given settings.

    Parameters
    ----------
    settings : dict
        Dictionary of settings to match against the models in the json file.
    json_data : dict
        json file which should contain an array models, with each model represented by a dictionary
        which contains one or more of the keys in InstrumentModelKeys.  The file that represents the
        model is specified by the value of the key InstrumentModelKeys.FILE.  The file name can point to
        different types of files, for example a .tif file for a saved PSF, or the directory containing
        the files for a deep learning model.
    """

    if "instrument_models" not in json_data:
        return None  # Ensure the "instrument_models" key is present

    instrument_models = json_data["instrument_models"]

    for instrument_model in instrument_models:
        # Exclude "settings_name" and "file" keys
        keys_to_check = [key for key in instrument_model.keys() if key not in [InstrumentModelKeys.MODEL_NAME.value, InstrumentModelKeys.FILE.value]]

        keys_to_check = set(instrument_model.keys()) & set(settings.keys())

        # Check if all non-excluded keys have the same values
        if all(instrument_model[key] == settings[key] for key in keys_to_check):
            return instrument_model[InstrumentModelKeys.FILE.value]  # Found a matching setting

    return None  # No matching setting found

def get_psf(path_, settings, json_data=None):    
    if json_data is None:
        json_data = load_instrument_models_json(path_)
        if json_data is None:
            return None
    psf_name = get_model_file(settings, json_data)
    if psf_name is None:
        return None
    else:
        return imread(os.path.join(path_, psf_name))

def load_instrument_models_json(path_):
    # check if path exists
    if not os.path.exists(path_):
        print("Path does not exist")
        return None

    # check if instrument_models.json exists
    if not os.path.exists(path_ + '/' + 'instrument_models.json'):
        print("instrument_models.json does not exist")
        return None

    # read json
    with open(path_ + '/' + 'instrument_models.json') as json_file:
        json_data = json.load(json_file)
        return json_data

def write_psf(path_, model, psf):
    
    # check if path exists
    if not os.path.exists(path_):
        os.makedirs(path_)

    # check if instrument_models.json exists
    if not os.path.exists(path_ + '/' + 'instrument_models.json'):
        # create instrument_models.json
        model[InstrumentModelKeys.FILE.value] = model[InstrumentModelKeys.MODEL_NAME.value] + '_psf.tif'
        with open(path_ + '/' + 'instrument_models.json', 'w') as outfile:
            json.dump({'instrument_models': [model]}, outfile)
        imsave(path_ + '/' + model[InstrumentModelKeys.FILE.value], psf)
    else:
        # append to existing instrument_models.json
        with open(path_ + '/' + 'instrument_models.json') as json_file:
            data = json.load(json_file)
            temp = data['instrument_models']
            set_unique_model_name(temp, model)
            model[InstrumentModelKeys.FILE.value] = model[InstrumentModelKeys.MODEL_NAME.value] + '_psf.tif'
            temp.append(model)
        with open(path_ + '/' + 'instrument_models.json', 'w') as outfile:
            json.dump(data, outfile)
        imsave(path_ + '/' + model['file'], psf)

def get_unique_name(names, name):
    i=1
    unique = False
    new_name = name

    while (unique == False):
        if new_name in names: 
            new_name = name + '_' + str(i)
            i = i + 1
        else:
            unique = True

    return new_name
    
def set_unique_model_name(instrument_models, model):
    name = model.get(InstrumentModelKeys.MODEL_NAME.value, None)

    if name is None:
        name = 'model'
    
    names = [instrument_model[InstrumentModelKeys.MODEL_NAME.value] for instrument_model in instrument_models]
    name = get_unique_name(names, name)
    
    model[InstrumentModelKeys.MODEL_NAME.value] = name
            