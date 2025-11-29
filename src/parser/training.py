import os

from .base import add_misc_options, add_cuda_options, adding_cuda, ArgumentParser
from .tools import save_args
from .dataset import add_dataset_options
from .model import add_model_options, parse_modelname
from .checkpoint import construct_checkpointname


def add_training_options(parser):
    group = parser.add_argument_group('Training options')
    group.add_argument("--batch_size", type=int, required=True, help="size of the batches")
    group.add_argument("--num_epochs", type=int, required=True, help="number of epochs of training")
    group.add_argument("--lr", type=float, required=True, help="AdamW: learning rate")
    group.add_argument("--snapshot", type=int, required=True, help="frequency of saving model/viz")
    group.add_argument("--use_mse_loss", action='store_true', help="Use MSE loss instead of L1 for reconstruction")
    group.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    

def parser():
    parser = ArgumentParser()

    # misc options
    add_misc_options(parser)

    # cuda options
    add_cuda_options(parser)
    
    # training options
    add_training_options(parser)

    # dataset options
    add_dataset_options(parser)

    # model options
    add_model_options(parser)

    opt = parser.parse_args()
    
    # remove None params, and create a dictionnary
    parameters = {key: val for key, val in vars(opt).items() if val is not None}

    # parse modelname
    ret = parse_modelname(parameters["modelname"])
    parameters["modeltype"], parameters["archiname"], parameters["losses"] = ret
    
    # update lambdas params
    lambdas = {}
    for loss in parameters["losses"]:
        lambdas[loss] = opt.__getattribute__(f"lambda_{loss}")
    parameters["lambdas"] = lambdas

    clip_lambdas = {'image':{}, 'text':{}, 'rendered_image':{}}
    
    # Process standard CLIP losses (image and text)
    for d in ['image', 'text']:  # Only process these, not 'rendered_image'
        losses_name = f'clip_{d}_losses'
        parameters[losses_name] = parameters[losses_name].split('_') if parameters[losses_name] != '' else []
        for loss in parameters[losses_name]:
            clip_lambdas[d][loss] = opt.__getattribute__(f"clip_lambda_{loss}")
            clip_lambdas[d][loss] = opt.__getattribute__(f"clip_lambda_{loss}")
    
    # Add visual CLIP loss (trajectory overlays) - special handling
    # NOTE: The code uses 'trajectory_overlays' key, not 'rendered_image'
    if hasattr(opt, 'clip_lambda_visual') and opt.clip_lambda_visual > 0:
        # Add 'trajectory_overlays' key for trajectory overlay features
        if 'trajectory_overlays' not in clip_lambdas:
            clip_lambdas['trajectory_overlays'] = {}
        clip_lambdas['trajectory_overlays']['cosine'] = opt.clip_lambda_visual
        print(f"✅ Trajectory overlays CLIP loss enabled: λ = {opt.clip_lambda_visual}")
    
    parameters["clip_lambdas"] = clip_lambdas

    if "folder" not in parameters:
        parameters["folder"] = construct_checkpointname(parameters, parameters["expname"])

    os.makedirs(parameters["folder"], exist_ok=True)
    save_args(parameters, folder=parameters["folder"])

    adding_cuda(parameters)
    
    return parameters
