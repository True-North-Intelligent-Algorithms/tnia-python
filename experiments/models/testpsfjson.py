import json
from tnia.deconvolution.psfs import gibson_lanni_3D, psf_from_beads
from tnia.plotting.projections import show_xy_zy_slice_center
import matplotlib.pyplot as plt
from tnia.models.instrument_models import write_psf, InstrumentModelKeys, InstrumentModalities, ModelTypes, get_model_file, load_instrument_models_json, get_psf
from skimage.io import imread
from tnia.deconvolution.psfs import psf_from_beads

print('test psf json')

psf_directory = r'D:/tests/deconvolution/psfs'

print(psf_directory)

xy_spacing = 0.1
z_spacing = 0.2
xy_size = 128
z_size = 64
NA = 1.4
ni = 1.515
ns = 1.4
emission_wavelength = 0.6
magnification = 100
depth = 0

theoretical_psf = gibson_lanni_3D(NA, ni , ns, xy_spacing, z_spacing, xy_size, z_size, depth, emission_wavelength, confocal = False, use_psfm=True)

beads_name = r"D:\\images\\tnia-python-images\\deconvolution\\beads\\Haase_Bead_Image1_crop.tif"
beads = imread(beads_name)
measured_psf, _, _=psf_from_beads(beads)

psf_directory = r'D:/tests/deconvolution/psfs'

settings1 = {
    InstrumentModelKeys.MODEL_NAME.value: 'set1',
    InstrumentModelKeys.XY_SPACING.value: 0.1,
    InstrumentModelKeys.Z_SPACING.value: 0.2,
    InstrumentModelKeys.XY_SIZE.value: 128,
    InstrumentModelKeys.Z_SIZE.value: 64,
    InstrumentModelKeys.NA.value: 1.4,
    InstrumentModelKeys.NI.value: 1.515,
    InstrumentModelKeys.NS.value: 1.4,
    InstrumentModelKeys.EMISSION_WAVELENGTH.value: 0.6,
    InstrumentModelKeys.MAGNIFICATION.value: 100,
    InstrumentModelKeys.DEPTH.value: 0,
    InstrumentModelKeys.MODALITY.value: InstrumentModalities.WIDEFIELD.value,
    InstrumentModelKeys.CONFOCAL_FACTOR.value: 1,
    InstrumentModelKeys.MODEL_TYPE.value: ModelTypes.THEORETICAL_PSF.value,
    InstrumentModelKeys.NOTES.value: 'theoretical PSF generated with psf models gibson lanni 3D',
}

settings1[InstrumentModelKeys.FILE.value] = settings1[InstrumentModelKeys.MODEL_NAME.value] + '.tif'

settings2 = {
    InstrumentModelKeys.MODEL_NAME.value: 'set2',
    InstrumentModelKeys.XY_SPACING.value: 0.2,
    InstrumentModelKeys.Z_SPACING.value: 0.4,
    InstrumentModelKeys.XY_SIZE.value: 256,
    InstrumentModelKeys.Z_SIZE.value: 512,
    InstrumentModelKeys.NA.value: 0.7,
    InstrumentModelKeys.NI.value: 1.515,
    InstrumentModelKeys.NS.value: 1.4,
    InstrumentModelKeys.EMISSION_WAVELENGTH.value: 0.6,
    InstrumentModelKeys.MAGNIFICATION.value: 100,
    InstrumentModelKeys.MODALITY.value: InstrumentModalities.WIDEFIELD.value,
    InstrumentModelKeys.CONFOCAL_FACTOR.value: 1,
    InstrumentModelKeys.DEPTH.value: 0,
    InstrumentModelKeys.MODEL_TYPE.value: ModelTypes.THEORETICAL_PSF.value,
    InstrumentModelKeys.NOTES.value: 'theoretical PSF generated with psf models gibson lanni 3D',
}

settings2[InstrumentModelKeys.FILE.value] = settings2[InstrumentModelKeys.MODEL_NAME.value] + '.tif'

settings3 = {
    InstrumentModelKeys.MODEL_NAME.value: 'Haase_Beads_PSF',
    InstrumentModelKeys.MODEL_TYPE.value: ModelTypes.MEASURED_PSF.value,
    InstrumentModelKeys.NOTES.value: 'measured PSF generated with Haase bead image'
}

settings3[InstrumentModelKeys.FILE.value] = settings3[InstrumentModelKeys.MODEL_NAME.value] + '.tif'


json_data=load_instrument_models_json(psf_directory)


write_psf(psf_directory, settings1, theoretical_psf)

theoretical_psf2 = gibson_lanni_3D(NA, ni , ns, xy_spacing, z_spacing, xy_size, z_size, depth, emission_wavelength, confocal = False, use_psfm=True)

write_psf(psf_directory, settings2, theoretical_psf2)

write_psf(psf_directory, settings3, measured_psf)

settings4 = {
    InstrumentModelKeys.MODEL_NAME.value: 'set4',
    InstrumentModelKeys.XY_SPACING.value: 0.1,
    InstrumentModelKeys.Z_SPACING.value: 0.2,
    InstrumentModelKeys.XY_SIZE.value: 128,
    InstrumentModelKeys.Z_SIZE.value: 64,
    InstrumentModelKeys.NA.value: 1.4,
    InstrumentModelKeys.NI.value: 1.515,
    InstrumentModelKeys.NS.value: 1.4,
    InstrumentModelKeys.EMISSION_WAVELENGTH.value: 0.6,
    InstrumentModelKeys.MAGNIFICATION.value: 100,
    InstrumentModelKeys.DEPTH.value: 100,
    InstrumentModelKeys.MODALITY.value: 'widefield',
    InstrumentModelKeys.CONFOCAL_FACTOR.value: 1,
    InstrumentModelKeys.MODEL_TYPE.value: 'psf_models_gibson_lanni_3D',
}

json_data = load_instrument_models_json(psf_directory)

print(get_model_file(settings1, json_data))
print(get_model_file(settings2, json_data))
print(get_model_file(settings3, json_data))
print(get_model_file(settings4, json_data))

psf_theoretical_back = get_psf(psf_directory, settings1)
psf_measured_back=get_psf(psf_directory, settings3)

fig = show_xy_zy_slice_center(psf_theoretical_back)
plt.show(block=True)
fig = show_xy_zy_slice_center(psf_measured_back)
plt.show(block=True)


'''
settings = {'settings': [settings1, settings2]}

with open(psf_directory + '/' + 'instrument_models.json', 'w') as outfile:
    # write settings1 and settings2 to test.json as array called settings
    json.dump(settings,outfile)

# read json
with open(psf_directory + '/' + 'test.json') as json_file:
    data = json.load(json_file)
    print('the first one')
    print(data['settings'][0])
    print()
    print('the second one')
    print(data['settings'][1])
'''
