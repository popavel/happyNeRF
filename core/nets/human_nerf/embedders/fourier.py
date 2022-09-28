import torch
import torch.nn as nn

class Embedder:
    def __init__(self, canonical_joints=None, **kwargs):
        self.kwargs = kwargs
        self.canonical_joints = canonical_joints
        self.create_embedding_fn_with_overparameterizing() if canonical_joints is not None else self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def create_embedding_fn_with_overparameterizing(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        eps = 0.00005
        for joint in self.canonical_joints:
            joint_tensor = torch.from_numpy(joint)

            # distances from the query point to the joints
            embed_fns.append(lambda x: torch.cdist(x, joint_tensor[None, ...].cuda(device=x.device)))
            out_dim += 1

            # TODO: move to Thesis:
            #  Positional encoding is not necessarily needed here, since we introduce overparameterization
            #  for more control over the general pose and shape. Positional encoding, however, helps to better
            #  learn high-frequency variation in color and geometry. See NeRF paper, section 5.1.

            # for freq in freq_bands:
            #     for p_fn in self.kwargs['periodic_fns']:
            #         embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(
            #             torch.cdist(x, joint_tensor[None, ...].cuda(device=x.device)) * freq))
            #         out_dim += 1

            # angles between the query point and the joints
            embed_fns.append(lambda x: torch.acos(torch.clamp(
                torch.div(
                    torch.matmul(x, joint_tensor[..., None].cuda(device=x.device)),
                    torch.add((torch.norm(x, dim=1) * torch.norm(joint_tensor.cuda(device=x.device)))[..., None], eps)
                ), min=-1+eps, max=1-eps)
            ))
            out_dim += 1

            # for freq in freq_bands:
            #     for p_fn in self.kwargs['periodic_fns']:
            #         embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(
            #             torch.acos(torch.clamp(
            #                 torch.div(
            #                     torch.matmul(x, joint_tensor[..., None].cuda(device=x.device)),
            #                     torch.add((torch.norm(x, dim=1) * torch.norm(joint_tensor.cuda(device=x.device)))[..., None], eps)
            #                 ), min=-1 + eps, max=1 - eps)
            #             ) * freq))
            #         out_dim += 1

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, canonical_joints=None):

    always_active = False
    if canonical_joints is None or (canonical_joints is not None and i == 0):
        always_active = True
    assert always_active

    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(canonical_joints=canonical_joints, **embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim
