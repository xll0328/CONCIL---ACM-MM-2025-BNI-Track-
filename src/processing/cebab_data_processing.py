import pandas as pd
import pickle
import os
import argparse
import yaml
from os.path import join
import random

attribute_map = {'Negative': 0, 'Positive': 1, 'unknown': 2, 'no majority': 2}

def read_csv_data(data_dir):

    cwd = os.getcwd()
    train1 = pd.read_csv(join(cwd, data_dir, "train_cebab_new_concept_single.csv"))
    train2 = pd.read_csv(join(cwd, data_dir, "train_yelp_exclusive_new_concept_single.csv"))

    val1 = pd.read_csv(join(cwd, data_dir, "dev_cebab_new_concept_single.csv"))
    val2 = pd.read_csv(join(cwd, data_dir, "dev_yelp_new_concept_single.csv"))

    test1 = pd.read_csv(join(cwd, data_dir, "test_cebab_new_concept_single.csv"))
    test2 = pd.read_csv(join(cwd, data_dir, "test_yelp_new_concept_single.csv"))

    train = pd.concat([train1, train2], axis = 0, ignore_index = True).fillna('unknown')
    val = pd.concat([val1, val2], axis = 0, ignore_index = True).fillna('unknown')
    test = pd.concat([test1, test2], axis = 0, ignore_index = True).fillna('unknown')
    attribute_name = train.columns[2:12] ## drop the last 3 attributes, same as the paper

    return train, val, test, attribute_name 

def create_new_data(data: pd.DataFrame):

    attribute_map = {'Negative': 0, 'Positive': 1, 'unknown': 2, 'no majority': 2}
    columns = data.columns
    text_column = columns[0]
    label_column = columns[1]
    new_data = []
    for i in range(len(data)):
        instance = data.iloc[i]
        new_instance, concept = {}, []

        if instance[label_column] != 'no majority':
            new_instance['text'] = instance[text_column]
            if isinstance(instance[label_column], str):
                label = int(instance[label_column])
            else:
                label = instance[label_column]
            new_instance['label'] =  label - 1  ## map[1, 2, 3, 4, 5] -> [0, 1, 2, 3, 4]
        
            for attribute in columns[2: 12]:
                concept.append(attribute_map[instance[attribute]])
            new_instance['concept'] = concept
            new_data.append(new_instance)
    
    return new_data

if __name__ == "__main__":
    
    random.seed(42)
    print('----------Processing CEBAB Dataset-----------')
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_dir', '-d', help='Where to save the new datasets')
    parser.add_argument('-data_dir', help='Where to load the datasets')
    args = parser.parse_args()
    data_dir = join(os.getcwd(), args.data_dir)
    save_dir = join(os.getcwd(), args.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_data, val_data, test_data, attribute_name = read_csv_data(args.data_dir)
    val_data = create_new_data(val_data)
    random.shuffle(val_data)
    split = int(len(val_data) * 0.5)

    for dataset in ['train', 'test']:
        print(f"Processing {dataset} set")
        f_name = dataset + '.pkl' 
        f = open(join(args.save_dir, f_name), 'wb')
        if 'train' in dataset:
            train_data = create_new_data(train_data) + val_data[:split]
            pickle.dump(train_data, f)
        else:
            test_data = create_new_data(test_data) + val_data[split:]
            pickle.dump(test_data, f)
        f.close()

    f = open(join(args.save_dir, 'attribute_name.pkl'), 'wb')
    pickle.dump(attribute_map, f)
    f.close()
    print('Done!')

    path = {
        'source_dir': data_dir,
        'processed_dir': save_dir
    }
    with open('src/utils/data_path.yml', 'a') as f:
        yaml.dump({'cebab': path}, f, default_flow_style = False)


        



