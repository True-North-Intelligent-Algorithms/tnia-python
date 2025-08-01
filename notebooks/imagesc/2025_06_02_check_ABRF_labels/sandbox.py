
def compute_lsa_distance2(ground, test):
    # Calculate the distance matrix and perform linear sum assignment

    M = ground.shape[0]
    N = test.shape[0]
    
    if M > N:
        # test has fewer points than ground
        dummy_cost = 1000000
        dummy_points = np.full((ground.shape[0] - test.shape[0], test.shape[1]), dummy_cost)
        test = np.vstack((test, dummy_points))
    elif N < M:
        # ground has fewer points than test
        dummy_cost = 1000000
        dummy_points = np.full((test.shape[0] - ground.shape[0], ground.shape[1]), dummy_cost)
        ground = np.vstack((ground, dummy_points))

    #test = test[:ground.shape[0]]  # Ensure test has the same number of points as ground
    
    dm = distance_matrix(ground, test)
    lsa = linear_sum_assignment(dm)

    lsa = list(lsa)

    # get rid of dummy points in the assignment
    '''
    for i, j in zip(lsa[0], lsa[1]):
        if i >= ground_original_shape[0] or j >= test_original_shape[0]:
            lsa[0] = np.delete(lsa[0], np.where(lsa[0] == i))
            lsa[1] = np.delete(lsa[1], np.where(lsa[1] == j))

            # remove i from ground and j from test
            ground = np.delete(ground, i, axis=0)
            test = np.delete(test, j, axis=0)
    '''

    # Filter out fake assignments
    matches = [
        (i, j) for i, j in zip(lsa[0], lsa[1])
        if i < M and j < N
    ]

    # convert matches back to numpy arrays
    lsa[0] = np.array([match[0] for match in matches])
    lsa[1] = np.array([match[1] for match in matches])

    displacement = ground[lsa[0]] - test[lsa[1]]
    distance = np.sqrt(np.sum(displacement**2, axis=1))

    lsa_mean = np.mean(distance)
    lsa_mse = np.mean(distance**2)

    return lsa[0], lsa[1], lsa_mse



def compute_lsa_distance2(ground, test):
    # Calculate the distance matrix and perform linear sum assignment
    test = test[:ground.shape[0]]  # Ensure test has the same number of points as ground
    dm = distance_matrix(ground, test)





    lsa = linear_sum_assignment(dm)
    

    n, m = len(ground), len(test)
    
    # Compute the real distance matrix
    dm = distance_matrix(ground, test)

    # Pad to square with large values (penalty for unassigned)
    if n > m:
        pad = np.full((n, n - m), dummy_cost)
        dm_padded = np.hstack((dm, pad))
    elif m > n:
        pad = np.full((m - n, m), dummy_cost)
        dm_padded = np.vstack((dm, pad))
    else:
        dm_padded = dm

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(dm_padded)

    # Filter out dummy matches (i.e., those matched to padded columns/rows)
    matched = [(i, j) for i, j in zip(row_ind, col_ind) if i < n and j < m]
    #displacement = np.array([ground[i] - test[j] for i, j in matched])
    #distances = np.linalg.norm(displacement, axis=1)    
    
    
    
    
    
    displacement = ground[lsa[0]] - test[lsa[1]]
    distance = np.sqrt(np.sum(displacement**2, axis=1))
    lsa_mean = np.mean(distance)
    lsa_mse = np.mean(distance**2)

    return lsa[0], lsa[1], lsa_mse

