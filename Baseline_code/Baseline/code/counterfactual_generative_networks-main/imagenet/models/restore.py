import torch
import os 

def restore_model(resume_epoch, ckpt_dir, Generator):
        """Restore the trained models and optimizers."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        path = os.path.join(ckpt_dir, 'model_18.tar'.format(resume_epoch))
        state_dict = torch.load(path, map_location=device)
        Generator.load_state_dict(state_dict['G'])
        print('The trained models from {} are loaded !'.format(path))