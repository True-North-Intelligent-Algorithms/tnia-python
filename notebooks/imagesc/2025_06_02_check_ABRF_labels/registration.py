# LMRG Study 4 Registration
#
# Author: Angel Mancebo (mance012@umn.edu)

import pathlib
import re
import time
from collections import namedtuple
from multiprocessing import Pool, freeze_support
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl
from pycpd import RigidRegistration
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from sklearn.neighbors import KDTree

registration_params = {
    'max_iterations': 1_000_000,
}
thread_params = {
    'limits': 1,
}
parallel_pool_params = {
    # Set 'processes' to fewer than the number of cores.
    'processes': 1,
    'initializer': None,
    'initargs': [],
    'maxtasksperchild': None,
}
pmap_params = {
    'chunksize': 1,
}

# ## Helper functions
# Define helper functions for extracting scale, angle, and translation from the registration results.

def get_scale(x) -> pd.Series:
    if x[1] is None:
        return pd.Series(np.nan)
    else:
        return pd.Series(x[1][0])


def get_angle(x) -> pd.Series:
    if x[1] is None:
        return pd.Series(np.nan)
    else:
        r = x[1][1]
        return pd.Series(
                np.arctan2(r[1, 0], r[0, 0])
        )


def get_translation(x) -> pd.Series:
    if x[1] is None:
        return pd.Series(np.array([np.nan, np.nan], dtype=np.float64))
    else:
        return pd.Series(x[1][2])


# ## Map file names
# Map the ground truth file names to the friendly file names used in the released images.

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


psf_map_xy = {  # micrometers, max measured (not theoretical)
    'fish': 0.290,
    'nuclei': None,
}
psf_map_z = {  # micrometers, measured (not theoretical)
    'fish': 0.182,
    'nuclei': None,
}

def register_guided(args: tuple[str, None]):
    f, _ = args

    results = {}
    results['filename'] = pathlib.Path(f).name

    # Store the coordinates of removed outliers.
    outliers = {}

    td_orig = test_data[test_data.loc[:, 'csv_path'] == f]
    gt_orig = ground_truth_coords[
        ground_truth_coords['ground_truth_name'] == td_orig['ground_truth_name'].iloc[0]
    ][['x', 'y', 'z']].values.astype('float64')
    gt = gt_orig.copy()

    # Initialize the result as an "empty" result.
    results['result'] = RegistrationResults(**{
        'test_coords_tf': None,
        'test_tf': None,
        'test_coords': td_orig,
        'ground_truth_coords': gt_orig,
        'outliers': outliers,
    })

    try:
        td_orig = td_orig[['x', 'y', 'z']].values.astype('float64')
        td = td_orig.copy()

        scale = gt[:, :2].std(axis=0).mean() / td[:, :2].std(axis=0).mean()
        translation = np.array([(td[:, :2].mean(axis=0) - gt[:, :2].mean(axis=0))])

        # Initialize a boolean array used to relax the outlier removal.
        select = np.array([False])
        # Keep track of registration iterations after removing outliers.
        iteration = 1

        category = re.match(r'.*(fish|nuclei).*', f.lower())

        flip_ids = ['R_3lAJ9xY4kGlL99f']
        for fid in flip_ids:
            if fid in f:
                td[:, :2] = np.hstack([td[:, 0].mean() - td[:, [0]], td[:, [1]]])

                td[:, :2] = (np.array([[0.0, 1.0],
                                    [-1.0, 0.0]]) @ td[:, :2].T).T

        while (~select).sum() > 0:
            reg = RigidRegistration(
                X=gt[:, :2],
                Y=td[:, :2],
                s=scale,
                t=translation,
                **registration_params,
            )
            test_coords_tf, test_tf = reg.register()

            # Put the z-direction back in to not have to keep track of the indices.
            test_coords_tf = np.hstack([test_coords_tf[:, :3], td[:, 2:3]]).copy()

            if 'nuclei' in f.lower():
                break

            dm = distance_matrix(gt[:, :2], test_coords_tf[:, :2])
            lsa = linear_sum_assignment(dm)
            distances = dm[lsa[0], lsa[1]]

            select = distances <= 2

            outliers[iteration] = test_coords_tf[lsa[1][~select], :]
            td = td[lsa[1][select], :]
            test_coords_tf = test_coords_tf[lsa[1][select], :]

            if category is not None:
                scale_xy = scales_map_xy[category.group(1)]

                # Use the scale parameter of the transformation to identify the
                # misaligned data. If the scale rounds to 1.0, don't use the
                # transformed coordinates.
                new_test_coords = td[:, :2].copy()
                if np.isclose(test_tf[0], 1.0, rtol=0.1, atol=0.02):
                    new_test_coords = new_test_coords + np.array([get_translation((test_coords_tf, test_tf))])
                elif np.isclose(test_tf[0], scale_xy, rtol=0.1, atol=0.02):
                    new_test_coords = scale_xy * new_test_coords
            
                td[:, :2] = new_test_coords.copy()
                iteration += 1

                results['result'] = RegistrationResults(**{
                    'test_coords_tf': test_coords_tf,
                    'test_tf': test_tf,
                    'test_coords': td,
                    'ground_truth_coords': gt,
                    'outliers': outliers,
                })

    except np.linalg.LinAlgError as lae:
        print(lae, pathlib.Path(f).relative_to(pathlib.Path(f).parent.parent))
        results['result'] = RegistrationResults(**{
            'test_coords_tf': None,
            'test_tf': None,
            'test_coords': td_orig,
            'ground_truth_coords': gt,
            'outliers': outliers,
        })

    except ValueError as ve:
        print(ve, pathlib.Path(f).relative_to(pathlib.Path(f).parent.parent))
        results['result'] = RegistrationResults(**{
            'test_coords_tf': None,
            'test_tf': None,
            'test_coords': td_orig,
            'ground_truth_coords': gt_orig,
            'outliers': outliers,
        })

    print(f'{pathlib.Path(f).relative_to(pathlib.Path(f).parent.parent)} has been registered.')
    return results

def check_if_close(x):
    # Original ground truth
    ratio = (ground_truth[x['category'].item()][x['dataset'].item()].values[:, 2].std() / 
    # Original z-coordinates (all)
    np.vstack(
        [
            x['result'].item().test_coords,
            *list(x['result'].item().outliers.values()),
        ],
    )[:, 2].std())
    # z scale in micrometers
    return pd.Series(
        [
            ratio,
            np.isclose(ratio, scales_map_z[x['category'].item()], rtol=0.1, atol=0.02),
            np.isclose(ratio, 1.0, rtol=0.1, atol=0.02),
            np.isclose(ratio, scales_map_z[x['category'].item()] / scales_map_xy[x['category'].item()], rtol=0.1, atol=0.02),
        ]
    )


def check_z_scale(x):
    ratio = (
        # Original ground truth
        ground_truth[x['category'].item()][x['dataset'].item()].values[:, 2].std() / 
        # Original z-coordinates (all)
        np.vstack(
            [
                x['result'].item().test_coords_tf,
                *list(x['result'].item().outliers.values()),
            ],
        )[:, 2].std())
    scales = [
        # Correct z scale
        1.0,
        # No scaling applied (pixel units)
        scales_map_z[x['category'].item()],
        # x-y scale applied instead of z
        scales_map_z[x['category'].item()] / scales_map_xy[x['category'].item()]
    ]
    for s in scales:
        if np.isclose(ratio, s, rtol=0.1, atol=0.02):
            print(s, 'isclose')
            rescaled_test_coords = x['result'].item().test_coords_tf.copy()
            # Only rescale z if the x-y scale or no scale was applied. If the correct scale was applied, don't touch.
            if s != scales[0]:
                rescaled_test_coords[:, 2] = rescaled_test_coords[:, 2] * s
                rescaled_test_coords[:, 2] += ground_truth[x['category'].item()][x['dataset'].item()].values[:, 2].mean() - rescaled_test_coords[:, 2].mean()
            return RegistrationResults(**{
                'test_coords_tf': x['result'].item().test_coords_tf,
                'test_tf': x['result'].item().test_tf,
                'test_coords': rescaled_test_coords,
                'ground_truth_coords': x['result'].item().ground_truth_coords,
                'outliers': x['result'].item().outliers,
            })
    return np.nan

def do_lsa(row):
    print(row[0])
    _, x = row
    series = pd.DataFrame.from_dict({'lsa': [linear_sum_assignment(
        distance_matrix(
            x['result'].ground_truth_coords, 
            x['result'].test_coords
        )
    )]})
    df = pd.concat([pd.DataFrame(np.array([x.values])), series], axis=1, ignore_index=True)
    df.columns = [*x.index, *series.columns]
    return df

def get_transformation(df):
    with threadpoolctl.threadpool_limits(**thread_params):
        with Pool(**parallel_pool_params) as p:
            tfs = p.map(_get_transformation_helper, iterrows_preserve_dtypes(df), **pmap_params)
    df_out = pd.concat(tfs)
    return df_out


def gen_raw_data(test_data) -> pd.DataFrame:
    raw_data_list = []
    for filename in test_data['filename'].unique():
        df = pd.DataFrame(columns=['filename', 'result', 'ground_truth_name'])
        test = test_data[test_data['filename'] == filename]
        df['filename'] = [filename]
        df['ground_truth_name'] = [test['ground_truth_name'].iloc[0]]
        ground = ground_truth_coords[ground_truth_coords['ground_truth_name'] == df['ground_truth_name'].iloc[0]]
        df['result'] = [RegistrationResults(
            test_coords_tf=test[['x', 'y', 'z']].dropna(axis=0, how='any').values,
            test_tf=None,
            test_coords=test[['x', 'y', 'z']].dropna(axis=0, how='any').values,
            ground_truth_coords=ground[['x', 'y', 'z']].dropna(axis=0, how='any').values,
            outliers={'0': []},
        )]
        if test[['x', 'y', 'z']].dropna(axis=0, how='any').shape[0] < 5:
            print(ground)
            print(test)
        raw_data_list.append(df)
    return pd.concat(raw_data_list, ignore_index=True)


def iterrows_preserve_dtypes(df):
    return ((i, df.loc[[i], :]) for i in df.index)


def nn_dist(df):
    """Returns mean and std of nn distribution."""
    means = []
    stdevs = []
    for _, row in iterrows_preserve_dtypes(df):
        try:
            ground = row['result'].item().ground_truth_coords
            test = row['result'].item().test_coords
            kd = KDTree(ground)
            distance, _ = kd.query(test, k=1, return_distance=True)
            means.append(np.mean(distance))
            stdevs.append(np.std(distance))
        except ValueError:
            means.append(np.nan)
            stdevs.append(np.nan)
    df_out = df.copy(deep=True)
    df_out['nn_mean'] = means
    df_out['nn_std'] = stdevs
    return df_out.drop(columns=['result', 'ground_truth_name', 'analysis_level']).copy(deep=True)


def lsa_dist_and_jaccard(df):
    """Returns mean and std of lsa distribution."""
    with threadpoolctl.threadpool_limits(**thread_params):
        with Pool(**parallel_pool_params) as p:
            dfs = p.map(_lsa_dist_and_jaccard_helper, iterrows_preserve_dtypes(df), **pmap_params)

    df_out = pd.concat(dfs)
    return df_out


def _lsa_dist_and_jaccard_helper(df_iter):
    """Helper function to enable data parallelism in `lsa_dist_and_jaccard`."""
    _, row = df_iter
    row = row.copy(deep=True)
    try:
        ground = row['result'].item().ground_truth_coords.astype('float64')
        test = row['result'].item().test_coords_tf.astype('float64')
        dm = distance_matrix(ground, test)
        lsa = linear_sum_assignment(dm)
        displacement = ground[lsa[0]] - test[lsa[1]]
        distance = np.sqrt(np.sum(displacement**2, axis=1))

        lsa_mean = np.mean(distance)
        lsa_std = np.std(distance)
        lsa_mse = np.mean(distance**2)
        row['lsa'] = [lsa]
        row['lsa_mean'] = lsa_mean
        row['lsa_std'] = lsa_std
        row['lsa_mse'] = lsa_mse

        category_match = re.match(r'(fish|nuclei)[1-4]', row['ground_truth_name'].item())
        if category_match is not None:
            category = category_match.group(1)
            j_distance = np.sqrt(np.sum((displacement / [[psf_map_xy[category]] * 2 + [psf_map_z[category]]])**2, axis=1))
            tp = np.sum(j_distance <= 1)
            fp = (test.shape[0] - ground.shape[0] if test.shape[0] >= ground.shape[0] else 0) \
            + sum(map(len, row['result'].item().outliers.values())) \
            + np.sum(j_distance > 1)
            fn = ground.shape[0] - test.shape[0] if test.shape[0] < ground.shape[0] else 0
            jac = tp / (tp + fp + fn)
            row['tp'] = [tp]
            row['fp'] = [fp]
            row['fn'] = [fn]
            row['jac'] = [jac]
        else:
            raise TypeError

    except (AttributeError, TypeError, ValueError):
        if 'lsa' not in locals():
            row['lsa'] = [None]

        for var in ['lsa_mean', 'lsa_std', 'lsa_mse', 'tp', 'fp', 'fn', 'jac']:
            if var in locals():
                row[var] = locals()[var]
            else:
                row[var] = [np.nan]

    return row.drop(columns=['result', 'ground_truth_name', 'analysis_level']).copy(deep=True)


def _get_transformation_helper(df_iter):
    _, row = df_iter
    flip_ids = ['R_3lAJ9xY4kGlL99f']
    try:
        ground = row['result'].item().ground_truth_coords.astype('float64')
        test = row['result'].item().test_coords.astype('float64')
        
        # Some data needs flipping. I chose to flip the y-direction
        # but it shouldn't matter after rotation from the
        # registration.
        for fid in flip_ids:
            if fid in str(row['filename'].item()):
                test = np.hstack([test[:, 0].mean() - test[:, [0]], test[:, [1]], test[:, [2]]])
                test = (np.array([[0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0]]) @ test.T).T
                break

        scale = (ground.std(axis=0) / test.std(axis=0))[:2].mean()
        translation = (ground.mean(axis=0) - test.mean(axis=0))[:2].reshape((1, 2))
        reg = RigidRegistration(
            X=ground[:, :2],
            Y=test[:, :2].copy(),
            s=scale,
            t=translation,
            **registration_params,
        )
        test_coords_tf, test_tf = reg.register()

        # Manually register the z coordinates using standard deviation.
        output_test_coords_tf = np.hstack([test_coords_tf, test[:, [2]]])
        # First, shift to the origin from the test center.
        output_test_coords_tf[:, 2] -= test[:, 2].mean()
        # Second, scale the coordinates to the ground truth scale.
        output_test_coords_tf[:, 2] *= ground[:, 2].std() / test[:, 2].std()
        # Third, shift to center of the ground truth.
        output_test_coords_tf[:, 2] += ground[:, 2].mean()
        result = RegistrationResults(
            test_coords_tf=output_test_coords_tf,
            test_tf=test_tf,
            test_coords=test,
            ground_truth_coords=ground,
            outliers={'0': []},
        )
    except (TypeError, ValueError) as error:
        print(error, row['filename'].item())
        result = RegistrationResults(
            test_coords_tf=None,
            test_tf=None,
            test_coords=row['result'].item().test_coords,
            ground_truth_coords=row['result'].item().ground_truth_coords,
            outliers={'0': []},
        )
        # tf = None
    row['result'] = [result]
    return row
 


if __name__ == '__main__':
    freeze_support()
    # ## Load data
    # Load the ground truth data.
    import os
    print()
    print(os.getcwd())
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(os.getcwd())
    ground_truth_coords = pd.read_csv(
        'ground_truth_coords.csv',
        index_col=0,
    )
    ground_truth_coords['ground_truth_name'] = (
        ground_truth_coords['path']
        .apply(lambda filename: friendly_names[pathlib.Path(filename).name])
    )


    # Load the test (submission) data and assigned it the ground truth by its friendly name.

    test_data = pd.read_csv('./coordinate_data_deidentified_nuclei.csv')
    test_data['filename'] = test_data['csv_path'].apply(lambda f: Path(f).name)
    for f, d in test_data.groupby('csv_path'):
        regex = re.match(r'.*((?:fish|nuclei)[1-4]).*', f.lower())
        if regex is not None:
            name = regex.group(1)
            test_data.loc[
                test_data.loc[:, 'csv_path'] == f, 'ground_truth_name'
            ] = name
        elif 'fish' in f.lower():
            regex = re.match(r'.*(Q1[0-4]\.13).*', f)
            if regex is not None:
                name = friendly_names[regex.group(1)]
                test_data.loc[
                    test_data.loc[:, 'csv_path'] == f, 'ground_truth_name'
                ] = name
        elif 'nuclei' in f.lower():
            regex = re.match(r'.*(Q[6-9]\.13).*', f)
            if regex is not None:
                name = friendly_names[regex.group(1)]
                test_data.loc[
                    test_data.loc[:, 'csv_path'] == f, 'ground_truth_name'
                ] = name
        else:
            print(f)


    with pd.option_context('display.max_colwidth', 1024):
        print(test_data[test_data['x'].isna()].dropna(axis=1, how='all'))


    # Ensure that each `ground_truth_name` has been assigned, that is, no `ground_truth_name` is empty. `True` if correct.

    for f in test_data['csv_path'].unique():
        if 'fish' not in f.lower():
            continue
        try:
            td_orig = test_data[test_data.loc[:, 'csv_path'] == f]
            gt_orig = ground_truth_coords[
                ground_truth_coords['ground_truth_name'] == td_orig['ground_truth_name'].iloc[0]
            ][['x', 'y', 'z']].values.astype('float32')
            td_orig = td_orig[['x', 'y', 'z']].values.astype('float32')
            if (re_match := re.match(r'(R_[0-9A-Za-z]+)_.*', pathlib.Path(f).name)) is not None:
                print(re_match.group(1))
        except ValueError as ve:
            print(ve, f)


    # ## Registration (only in x-y)
    # Perform rigid registration (with scale) on all the data sets, then extract the parameters.

    results = {'result': [], 'filename': []}
    scales_map_xy = {  # micrometers
        'fish': 0.1616,
        'nuclei': 0.124,
    }
    scales_map_z = {  # micrometers
        'fish': 0.200,
        'nuclei': 0.200,
    }


    RegistrationResults = namedtuple('RegistrationResults', [
        'test_coords_tf',
        'test_tf',
        'test_coords',
        'ground_truth_coords',
        'outliers',
    ])


    t0 = time.time()
    # threadpool_limits context must go AROUND the Pool context to limit threads.
    '''
    with threadpoolctl.threadpool_limits(**thread_params):
        with Pool(**parallel_pool_params) as p:
            results_map = p.map(
                register_guided, 
                test_data.groupby('csv_path'),
                **pmap_params,
            )
    '''
    results_map = [
        register_guided(temp)
        for temp in test_data.groupby('csv_path')
    ]
    print(time.time() - t0, 's')
    results_df = pd.DataFrame(columns=['result', 'filename'])
    for i, r in enumerate(results_map):
        results_df.loc[i, 'result'] = r['result']
        results_df.loc[i, 'filename'] = r['filename']

    results_df['result'].apply(get_angle)

    results_df.loc[:, 'scale'] = results_df['result'].apply(get_scale)
    results_df.loc[:, ['angle_z']] = np.vstack(
        results_df.loc[:, 'result'].apply(get_angle).values.tolist()
    )
    results_df.loc[:, ['translation_x', 'translation_y']] = np.vstack(
        results_df.loc[:, 'result'].apply(get_translation).values.tolist()
    )

    # ## Statistics
    # Get statistics on the results grouped but anonymized submitter names.

    results_summary = results_df.drop('result', axis=1)

    results_summary['id'] = (
        results_summary['filename']
        .str.extract(r'^(R_[a-zA-Z0-9]+)_.*')
    )
    results_summary.drop('filename', axis=1).groupby('id').agg(['mean', 'std'])


    # Perform linear sum assignment on the registration results, with test points sorted by increasing distance to nearest neigbor in the ground truth. In the case where the scale is close enough to 1.0, don't use the transformed test coordinates and instead use the original submitted coordinates.

    t0 = time.time()
    distances = {}
    lsa_indices = {}
    for filename in results_df.dropna(subset='scale')['filename']:
        test_coords_tf, test_tf, test_coords, gt, outliers = results_df.query(
            'filename == @filename'
        )['result'].iloc[0]
        kd = KDTree(gt[:, :2])
        td = test_coords
        pattern = re.match(r'.*(fish|nuclei).*', filename.lower())
        if pattern is not None:
            scale_xy = scales_map_xy[pattern.group(1)]
        else:
            continue
        distance, index = kd.query(test_coords_tf[:, :2], k=1, return_distance=True)
        distances[filename] = distance
        dm = distance_matrix(gt[:, :2], test_coords_tf[:, :2])
        lsa_indices[filename] = {}
        lsa_indices[filename]['dm'] = dm
        lsa_indices[filename]['lsa'] = linear_sum_assignment(dm)
    print(time.time() - t0, 's')


    results_df.iloc[[0], :]['result'].item()

    # Instead of doing this, just add the category to another column.
    results_df_success = results_df.drop(results_df[results_df['result'].apply(lambda x: x.test_coords_tf is None)].index)
    results_df_success = pd.concat(
        [
            results_df_success, 
            results_df_success['filename'].str.extract(r".*(?P<dataset>(?P<category>fish|nuclei)[1-4]).*"),
        ],
        axis=1,
    )

    ground_truth = {
        category:
        {
            dataset: data for (dataset, _), data in ground_truth_coords.groupby(['ground_truth_name', 'path']) if category in dataset
        }
        for category in ['fish', 'nuclei']
    }

    # ## Registration (in z)

    # Calculate the ratio of standard deviations between ground truth and test data.


    with pd.option_context('display.max_rows', 150):
        print(
            results_df_success
            .groupby(
                ['category', 'dataset', 'filename']
            )
            .apply(check_if_close)
        )


    # The point-spread function sizes are

    psf_map_xy = {  # micrometers, max measured (not theoretical)
        'fish': 0.290,
        'nuclei': None,
    }
    psf_map_z = {  # micrometers, measured (not theoretical)
        'fish': 0.182,
        'nuclei': None,
    }

    with pd.option_context('display.max_rows', 150, 'display.width', 1024):
        rescaled_results = (
            results_df_success
            .groupby(
                ['category', 'dataset', 'filename']
            )
            .apply(lambda x: pd.Series([check_z_scale(x)]))
            #.reset_index()
        )
        # Change the column name from 0 to 'result'
        rescaled_results.columns = [*rescaled_results.columns[:-1], 'result']
        print(rescaled_results)


    with Pool(**parallel_pool_params) as p:
        lsa = pd.concat(
            p.map(do_lsa, rescaled_results.dropna().query('category == "fish"').iterrows()),
            ignore_index=True,
        )

    # Raw data. This data has **not** been registered.

    raw_data = gen_raw_data()
    raw_data['analysis_level'] = 'raw_data'
    jac = lsa_dist_and_jaccard(raw_data)

    nn = nn_dist(raw_data)

    raw_data_results = raw_data.join(jac.set_index('filename'), on='filename').join(nn.set_index('filename'), on='filename')
    raw_data_results['scale_xy'] = raw_data_results['result'].apply(get_scale)
    raw_data_results[['translation_x', 'translation_y']] = raw_data_results['result'].apply(get_translation)
    raw_data_results['angle_xy'] = raw_data_results['result'].apply(get_angle)

    # Add um units
    for units, columns in zip(
            ['um', 'um^2', 'px'],
            [
                ['lsa_mean', 'lsa_std', 'nn_mean', 'nn_std'],
                ['lsa_mse'],
            ],
    ):
        for col in columns:
            if col in raw_data_results.columns:
                raw_data_results[col + ' (' + units + ')'] = raw_data_results[col].copy()
                raw_data_results.drop(col, axis=1, inplace=True)

        

    fully_transformed_data = get_transformation(raw_data.copy(deep=True))
    fully_transformed_data['analysis_level'] = 'fully_transformed'

    nn_full = nn_dist(fully_transformed_data)

    lsa_full = lsa_dist_and_jaccard(fully_transformed_data)

    fully_transformed_results = fully_transformed_data.join(lsa_full.set_index('filename'), on='filename').join(nn_full.set_index('filename'), on='filename')
    fully_transformed_results['scale_xy'] = fully_transformed_results['result'].apply(get_scale)
    fully_transformed_results[['translation_x', 'translation_y']] = fully_transformed_results['result'].apply(get_translation)
    fully_transformed_results['angle_xy'] = fully_transformed_results['result'].apply(get_angle)

    # Add units to column values that should have real units
    for units, columns in zip(
            ['um', 'um^2', 'px'],
            [
                ['lsa_mean', 'lsa_std', 'nn_mean', 'nn_std'],
                ['lsa_mse'],
            ],
    ):
        for col in columns:
            if col in fully_transformed_results.columns:
                fully_transformed_results[f'{col} ({units})'] = fully_transformed_results[col].copy()
                fully_transformed_results.drop(col, axis=1, inplace=True)

    output_stats = pd.concat(
        [raw_data_results, fully_transformed_results],
        ignore_index=True,
    )
    output_stats['id'] = output_stats['filename'].str.extract(r'(R_[0-9A-Za-z]+).*')

    # EXPORTED_RESULTS
    output_stats.drop(columns=['filename', 'result', 'lsa']).to_csv(
        'registration_stats.csv', index=False)
