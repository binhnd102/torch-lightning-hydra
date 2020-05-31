import data as data

def build_dataset(ds_name, data_dir, valid_size=0.3):
    return getattr(data, 'build_{}'.format(ds_name))(data_dir, valid_size)