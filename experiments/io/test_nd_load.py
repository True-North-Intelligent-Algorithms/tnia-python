import tnia.io.bioformats_helper as bfh
import bioformats

bfh.start_jvm()

#filename='D:\\elephasbio\\large montage\\720x AtlasVolume-05042021-0833-006\\720x AtlasVolume-05042021-0833-006.xml'
filename='D:\\images\\From Joao Mamede\\F2_laminigfpcarub8am.nd2'


meta=bioformats.get_omexml_metadata(filename)
o=bioformats.OMEXML(meta)

nz = o.image().Pixels.SizeZ