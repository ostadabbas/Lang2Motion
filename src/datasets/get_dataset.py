from .amass import AMASS
from .mevis_dataset import MeVisDataset

def get_dataset(name="amass"):
    if name == "mevis":
        return MeVisDataset
    return AMASS


def get_datasets(parameters, clip_preprocess, split="train"):
    dataset_name = parameters.get("dataset", "amass")
    if dataset_name == "mevis":
        DATA = MeVisDataset
    else:
        DATA = AMASS

    # Remove split from parameters to avoid duplicate argument
    params_copy = parameters.copy()
    params_copy.pop('split', None)
    
    if split == 'all':
        train = DATA(split='train', clip_preprocess=clip_preprocess, **params_copy)
        test = DATA(split='val', clip_preprocess=clip_preprocess, **params_copy)

        # add specific parameters from the dataset loading
        train.update_parameters(parameters)
        test.update_parameters(parameters)
    else:
        dataset = DATA(split=split, clip_preprocess=clip_preprocess, **params_copy)
        train = dataset

        # For memory efficiency, don't load a separate validation set during training
        # Just use a reference to the training set (validation not used in training loop anyway)
        test = train

        # add specific parameters from the dataset loading
        dataset.update_parameters(parameters)

    datasets = {"train": train,
                "test": test}

    return datasets
