def apply_corrections(data, responseid, data_name):
    
    data_corrected = data.copy()
    correction = "NONE"
    
    if 'nuclei' in data_name.lower():
        
        if responseid in ['R_____2rCMx6wAGE7bFJh___kj____']:
            # z offset by 4
            if data_name == 'nuclei1':
                data_corrected['z'] = data['z'] - 4

                correction = 'Z_OFFSET_BY_4'
            elif data_name in ['nuclei2', 'nuclei3']:
                # z offset by 10
                data_corrected['z'] = data['z'] - 10
                correction = 'Z_OFFSET_BY_10'
            elif data_name == 'nuclei4':
                # z offset by 6
                data_corrected['x'] = data['y']*(.9)
                data_corrected['y'] = data['x']*(.9)
                data_corrected['z'] = data['z']*(1.05)
                data_corrected['z'] = data['z'] - 20
                correction = 'Z_OFFSET_BY_6'

        elif responseid in ['R_31bjqd6Mm8wBxN5']:
            # reverse x and y
            data_corrected['x'] = data['y']
            data_corrected['y'] = data['x']
            data_corrected['z'] = data['z']
            correction = 'REVERSE_XY' 
        
        elif responseid in ['R_3lAJ9xY4kGlL99f']:

            data_corrected['x'] = data['y']*(4.)
            data_corrected['y'] = data['x']*(4.)
            data_corrected['z'] = data['z']*(4.)

            correction = 'REVERSE_XY_SCALE4'        
        
        elif responseid in ['R_0638fJAPpmzLtu1', 'R_3RxOa8kaiyul4bG']:
            print('applying spaces', responseid)
            data_corrected['x'] = data['x']*(.124)
            data_corrected['y'] = data['y']*(.124)
            data_corrected['z'] = data['z']*(.2)
            
            correction = 'PIXELS_TO_MICRONS'

    else: # For fish datasets
        if responseid in ['R_3lAJ9xY4kGlL99f']:
            
            # reverse x and y
            data_corrected['x'] = data['y']
            data_corrected['y'] = data['x']
            data_corrected['z'] = data['z']
        
            correction = 'FLIP_XY'

        elif responseid in ['R_1q7MrEdSr6yhykH', 'R_24HIjcCJh6uI3bu', 'R_31bjqd6Mm8wBxN5', 'R_3RxOa8kaiyul4bG']:
            # pixels to microns
            data_corrected['x'] = data['x']*(.1616160)
            data_corrected['y'] = data['y']*(.1616160)
            data_corrected['z'] = data['z']*(.2)

            correction = 'PIXELS_TO_MICRONS'
        elif responseid in ['R_2c14tLfUPR1Vnua']:
            data_corrected['z'] = data['z']*(.2)
            correction = 'Z_PIXELS_TO_MICRONS'
        elif responseid in ['R_2DNRFrAvCDUX1EL']:
            if data_name == 'fish1' or data_name == 'fish2' or data_name == 'fish4':
                # z offset by 4
                #data_corrected['z'] = data['z'] - 4
                #correction = 'Z_OFFSET_BY_4'
                data_corrected['z'] = data['z']*(.001)
                correction = 'Z_NANOS_TO_MICRONS'
        elif responseid in ['R_2cCJjlMU7i9XjMQ']:
            data_corrected['z'] = data['z']*(.1616160)
            correction = 'Z_PIXELS_SCALE_BY_1616160'
        elif responseid in ['R_1qdHCwPCdNvs9xi']:
            data_corrected['z'] = data['z']*(.2/ .1616160)
            correction = 'Z_PIXELS_1616160_to_2'
        
    return data_corrected, correction