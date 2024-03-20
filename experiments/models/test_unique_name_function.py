from tnia.models.instrument_models import get_unique_name, set_unique_model_name, InstrumentModelKeys

name = 'set1'

names = ['set1_2','set1', 'set2', 'set3']

new_name = get_unique_name(names, name)

print(name, new_name)

name = 'set5'
new_name = get_unique_name(names, name)
print(name, new_name)

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
    InstrumentModelKeys.MODALITY.value: 'widefield',
    InstrumentModelKeys.CONFOCAL_FACTOR.value: 1,
    InstrumentModelKeys.MODEL_TYPE.value: 'psf_models_gibson_lanni_3D',
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
    InstrumentModelKeys.MODALITY.value: 'widefield',
    InstrumentModelKeys.CONFOCAL_FACTOR.value: 1,
    InstrumentModelKeys.DEPTH.value: 0,
    InstrumentModelKeys.MODEL_TYPE.value: 'psf_models_gibson_lanni_3D',
}

settings2[InstrumentModelKeys.FILE.value] = settings2[InstrumentModelKeys.MODEL_NAME.value] + '.tif'

settings3 = {
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
    InstrumentModelKeys.MODALITY.value: 'widefield',
    InstrumentModelKeys.CONFOCAL_FACTOR.value: 1,
    InstrumentModelKeys.MODEL_TYPE.value: 'psf_models_gibson_lanni_3D',
}

print('name before adjustment',settings3[InstrumentModelKeys.MODEL_NAME.value])

set_unique_model_name([settings1, settings2], settings3)

print('name after adjustment',settings3[InstrumentModelKeys.MODEL_NAME.value])