import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

friendly_names = {
    'celegans_dyn-90_ceff-0_label.ics.ome.tiff': 'fish1',
    'celegans_dyn-90_ceff-90_label.ics.ome.tiff': 'fish2',
    'celegans_dyn-10_ceff-0_label.ics.ome.tiff': 'fish3',
    'celegans_dyn-10_ceff-90_label.ics.ome.tiff': 'fish4',

    'Q10.13': 'fish1',
    'Q11.13': 'fish2',
    'Q12.13': 'fish3',
    'Q13.13': 'fish4',

    'out_c00_dr10_label.tif': 'nuclei3',
    'out_c00_dr90_label.tif': 'nuclei1',
    'out_c90_dr10_label.tif': 'nuclei4',
    'out_c90_dr90_label.tif': 'nuclei2',

    'Q6.13': 'nuclei1',
    'Q7.13': 'nuclei2',
    'Q8.13': 'nuclei3',
    'Q9.13': 'nuclei4',

    }

def get_friendly_name(filename):
    """
    Returns a friendly name for the given filename.
    If the filename is not in the dictionary, it returns the filename itself.
    """
    return friendly_names.get(filename, filename)

def find_scale_and_translation(ground, test, use_lsa=True):
    ground_ = ground[['x', 'y', 'z']].values
    test_ = test[['x', 'y', 'z']].values

    if use_lsa:
        a, b, lse_mse = compute_lsa_distance(ground_, test_)

        ground_ = ground_[a]
        test_ = test_[b]

    scalexy = (ground_.std(axis=0) / test_.std(axis=0))[:2].mean()
    test_[:,:2] = test_[:,:2]* scalexy
    translationxy = (ground_.mean(axis=0) - test_.mean(axis=0))[:2].reshape((1, 2))

    scalez = ground_[:, 2].std() / test_[:, 2].std()
    test_[:, 2] = test_[:, 2] * scalez
    translationz = (ground_[:, 2].mean() - test_[:, 2].mean()).reshape((1, 1))
    translation = np.hstack((translationxy, translationz))
    scale = np.array([[scalexy, scalexy, scalez]])
    return scale, translation

def compute_lsa_distance(ground, test):
    # Calculate the distance matrix and perform linear sum assignment
    dm = distance_matrix(ground, test)
    lsa = linear_sum_assignment(dm)
    
    displacement = ground[lsa[0]] - test[lsa[1]]
    distance = np.sqrt(np.sum(displacement**2, axis=1))

    lsa_mse = np.mean(distance**2)

    return lsa[0], lsa[1], lsa_mse

psf_map_xy = {  # micrometers, max measured (not theoretical)
    'fish': 0.290,
    'nuclei': None,
}
psf_map_z = {  # micrometers, measured (not theoretical)
    'fish': 0.182,
    'nuclei': None,
}

def calculate_jac(ground, test, lsa=None, category='fish'):
    
    displacement = ground[lsa[0]] - test[lsa[1]]
    distance = np.sqrt(np.sum(displacement**2, axis=1))
    j_distance = distance #np.sqrt(np.sum((displacement / [[psf_map_xy[category]] * 2 + [psf_map_z[category]]])**2, axis=1))
    j_distance = np.sqrt(np.sum((displacement / [[psf_map_xy[category]] * 2 + [psf_map_z[category]]])**2, axis=1))
    tp = np.sum(j_distance <= 1)
    fp = (test.shape[0] - ground.shape[0] if test.shape[0] >= ground.shape[0] else 0) + np.sum(j_distance > 1)
    #+ sum(map(len, row['result'].item().outliers.values())) \
    fn = ground.shape[0] - test.shape[0] if test.shape[0] < ground.shape[0] else 0
    jac = tp / (tp + fp + fn)

    return tp, fp, fn, jac