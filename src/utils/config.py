CONFIG = {
    # CUB Dataset
    'cub': {
        'N_CONCEPTS': 116,
        'N_CLASSES': 200,
        'trigger_value': 1
    },

    # AwA Dataset
    'awa': {
        'N_CONCEPTS': 85,
        'N_CLASSES': 50,
        'trigger_value': 1
    },

    # CEBAB Dataset
    'cebab': {
        'N_CONCEPTS': 10,
        'N_CLASSES': 5,
        'N_CONCEPTS_CLASSES': 3,
        'trigger_value': [(0, 1), (1, 1), (3, 0), (4, 0)]   
    },

    # IMDB Dataset
    'imdb': {
        'N_CONCEPTS': 8,
        'N_CLASSES': 2,
        'N_CONCEPTS_CLASSES': 3,
        'trigger_value': [(0, 1), (1, 0)]
    }
}


