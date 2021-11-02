import microscPSF.microscPSF as msPSF

def gibson_lanni_3D(NA, ni, ns, pixel_size, xy_size, zv, pz):
    m_params = msPSF.m_params
    m_params['NA']=NA
    m_params['ni']=ni
    m_params['ni0']=ni
    m_params['ns']=ns

    return msPSF.gLXYZFocalScan(m_params, pixel_size, xy_size, zv, pz)
