polystr_to_idx = {'2a11': 0, '2a9': 1, '3a11': 2, '2a7': 3, '3a10': 4, '4a11': 5, '3a8': 6, '2a5': 7, '5a12': 8, '3a7': 9, '4a9': 10, '5a11': 11, '6a11': 12, '5a9': 13, '4a7': 14, '7a12': 15, '3a5': 16, '5a8': 17, '7a11': 18, '2a3': 19, '7a10': 20, '5a7': 21, '8a11': 22, '3a4': 23, '7a9': 24, '4a5': 25, '9a11': 26, '5a6': 27, '6a7': 28, '7a8': 29, '8a9': 30, '9a10': 31, '10a11': 32, '11a12': 33}

idx_to_polystr = {i:p for (p,i) in polystr_to_idx.items()}

num_poly = len(polystr_to_idx)

class_arr = [idx_to_polystr[i] for i in range(num_poly)]
