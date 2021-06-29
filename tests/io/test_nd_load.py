import tnia.io.bioformats_helper as bfh
import bioformats

bfh.start_jvm()

filename='D:\\elephasbio\\large montage\\720x AtlasVolume-05042021-0833-006\\720x AtlasVolume-05042021-0833-006.xml'

meta=bioformats.get_omexml_metadata(filename)
o=bioformats.OMEXML(meta)

nz = o.image().Pixels.SizeZ