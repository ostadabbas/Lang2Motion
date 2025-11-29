from src.models.architectures.transformer import Encoder_TRANSFORMER, Decoder_TRANSFORMER, Decoder_MLP
from src.models.modeltype.motionclip import MOTIONCLIP

JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices"]

LOSSES = ["rc", "rcxyz", "vel", "velxyz", "range", "veldist", "diversity", "spatial", "textrecon"]  # Enhanced motion-preserving losses added

def get_model(parameters, clip_model):
    # Pass CLIP model to encoder so it can use overlayed images
    encoder_params = parameters.copy()
    encoder_params['clip_model'] = clip_model
    encoder = Encoder_TRANSFORMER(**encoder_params)
    
    # Choose decoder type based on parameter
    use_mlp_decoder = parameters.get('mlp_decoder', False)
    
    if use_mlp_decoder:
        use_cross_attention = parameters.get('use_cross_attention', False)
        # Create decoder params without use_cross_attention to avoid duplicate
        decoder_params = {k: v for k, v in parameters.items() if k != 'use_cross_attention'}
        decoder = Decoder_MLP(use_cross_attention=use_cross_attention, **decoder_params)
        if use_cross_attention:
            print("Using MLP Decoder with 77-token cross-attention")
        else:
            print("Using MLP Decoder (no motion compression)")
    else:
        decoder = Decoder_TRANSFORMER(**parameters)
        print("Using Transformer Decoder")
    
    parameters["outputxyz"] = "rcxyz" in parameters["lambdas"]
    return MOTIONCLIP(encoder, decoder, clip_model=clip_model, **parameters).to(parameters["device"])
