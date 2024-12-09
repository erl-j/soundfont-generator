import torch
import pytorch_lightning as pl
from torch import nn
from tqdm import tqdm
import numpy as np
import einops
import wandb
import torch
# import wandb logging
from pytorch_lightning.loggers import WandbLogger
from stable_audio_tools import get_pretrained_model
from transformers import T5Tokenizer, T5EncoderModel


class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
    
class FourierFeatures(nn.Module):

    def __init__(self, in_features, out_features, n_layers):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        layers = []
        layers += [nn.Linear(in_features, out_features)]
        # add sin activation
        layers += [SinActivation()]
        for i in range(n_layers-1):
            layers += [nn.Linear(out_features, out_features)]
            layers += [SinActivation()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class FlowMatchingModule(pl.LightningModule):

    def __init__(self, main_model=None, text_conditioner=None, max_tokens=128, n_channels=None, t_input=None):
        super().__init__()
        self.save_hyperparameters(ignore=['main_model', "text_conditioner"])

        self.model = main_model.transformer
        self.input_layer = main_model.transformer.project_in
        self.output_layer = main_model.transformer.project_out

        self.text_conditioner = text_conditioner

        self.d_model = self.input_layer.weight.shape[0]
        self.d_input = self.input_layer.weight.shape[1]

        # use fourier features for schedule
        self.schedule_embedding = FourierFeatures(1, self.d_model, 2)
        # use learned positional encoding
        self.pitch_embedding = nn.Parameter(torch.randn(n_channels, self.d_model))
        # make embedding layer for tags
        self.channels =  n_channels

        mean_proj = []
        for layer in self.model.layers:
            mean_proj += [nn.Linear(self.d_model, self.d_model)]
        self.mean_proj = nn.ModuleList(mean_proj)

    def get_example_inputs(self):
        text = "A piano playing a C major chord"
        conditioning, conditioning_mask = self.text_conditioner(text, device = self.device)

        # repeat conditioning
        conditioning = einops.repeat(conditioning, 'b t d-> b t c d', c=self.channels)
        conditioning_mask = einops.repeat(conditioning_mask, 'b t -> b t c', c=self.channels)

        t = torch.rand(1, device=self.device)
        z = torch.randn(1, self.hparams.t_input ,self.hparams.n_channels, self.d_input , device=self.device)
        return z, conditioning, conditioning_mask, t


    def forward(self, x, conditioning, conditioning_mask, t):
        batch, t_input, n_channels, d_input = x.shape

        # add conditioning to x
        x = self.input_layer(x)
        tz = self.schedule_embedding(t[:,None,None,None])
        pitch_z = self.pitch_embedding[None, None, :n_channels, :]
        # print shapes
        x = x + tz + pitch_z
        rot = self.model.rotary_pos_emb.forward_from_seq_len(x.shape[1])

        conditioning = einops.rearrange(conditioning, 'b t c d -> (b c) t d', c=self.channels)
        conditioning_mask = einops.rearrange(conditioning_mask, 'b t c -> (b c) t', c=self.channels)
        
        for layer_idx, layer in enumerate(self.model.layers):
            x = einops.rearrange(x, 'b t c d -> (b c) t d')
            
            x = layer(x, rotary_pos_emb=rot, context = conditioning, context_mask = conditioning_mask)
            x = einops.rearrange(x, '(b c) t d -> b t c d', c=self.channels)
            x_ch_mean = x.mean(dim=2)
            x_ch_mean = self.mean_proj[layer_idx](x_ch_mean)
            # non linearity
            # x_ch_mean = torch.relu(x_ch_mean)
            # # layer norm
            # x_ch_mean = torch.layer_norm(x_ch_mean, x_ch_mean.shape[1:])
            x += x_ch_mean[:, :, None, :]
        x = self.output_layer(x)
        return x
        
    def step(self, batch, batch_idx):
        x = batch["z"]
        text = batch["text"]
        conditioning, conditioning_mask = self.text_conditioner(text, device = self.device)

        # repeat conditioning
        conditioning = einops.repeat(conditioning, 'b t d-> b t c d', c=self.channels)
        conditioning_mask = einops.repeat(conditioning_mask, 'b t -> b t c', c=self.channels)

        x = einops.rearrange(x, 'b c d t -> b t c d')
        z0 = torch.randn(x.shape, device=x.device)
        z1 = x
        t = torch.rand(x.shape[0], device=x.device)
        zt = t[:,None,None,None] * z1 + (1 - t[:,None,None,None]) * z0
        vt = self(zt,conditioning,conditioning_mask,t)
        loss = (vt - (z1 - z0)).pow(2).mean()
        return loss
    
    @torch.inference_mode()
    def sample(self, batch_size, text, steps=10, same_latent=False):
        # Ensure model is on the correct device
        device = next(self.parameters()).device
        dtype = self.input_layer.weight.dtype

        # Move conditioning to the correct device and dtype
        conditioning, conditioning_mask = self.text_conditioner(text, device=device)
        conditioning = einops.repeat(conditioning, "b t d-> b t c d", c=self.channels)
        conditioning_mask = einops.repeat(
            conditioning_mask, "b t -> b t c", c=self.channels
        )
        conditioning = conditioning.to(device=device, dtype=dtype)
        conditioning_mask = conditioning_mask.to(device=device)

        self.eval()
        with torch.no_grad():
            # Create initial noise on the correct device and dtype
            z0 = torch.randn(
                batch_size,
                self.hparams.t_input,
                self.hparams.n_channels,
                self.d_input,
                device=device,
                dtype=dtype,
            )

            if same_latent:
                z0 = z0[0].repeat(batch_size, 1, 1, 1)

            zt = z0
            for step in tqdm(range(steps)):
                t = torch.tensor([step / steps], device=device, dtype=dtype)
                zt = zt + (1 / steps) * self.forward(
                    zt, conditioning, conditioning_mask, t
                )

            return zt

        
    def training_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('trn_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)
    
class EncodedAudioDataset(torch.utils.data.Dataset):
    def __init__(self, paths, pitch_range):
        records = []
        print("Loading data")
        for path in tqdm(paths):
            records+=torch.load(path)
        self.records = records
        self.pitch_range = pitch_range

        # keep only records with z
        self.records = [r for r in self.records if "z" in r]

        print(f"Loaded {len(self.records)} records")
   

    def compose_prompt(self,record):
        title = record["name"] if "name" in record else record["title"]
       
        tags = record["tags"]

        # take tags
        # shuffle
        tags = np.random.choice(tags, len(tags), replace=False)
        # take random number of tags
        tags = list(tags[:np.random.randint(0, len(tags)+1)])
        # 
        # take either the title or group or type or nothing
        if "type_group" in record and "type" in record:
            type_group = record["type_group"]
            type = record["type"]
            head = np.random.choice([title, type_group, type])
        else:
            head = np.random.choice([title])

        # append tags
        # with 75% chance add head
        elements = tags
        if np.random.rand() < 0.75:
            elements = [head] + elements

        # shuffle elements
        elements = np.random.choice(elements, len(elements), replace=False)

        prompt = " ".join(elements)
        
        # make everything lowercase
        prompt = prompt.lower()
        return prompt

    def __len__(self):
        return len(self.records)
    
    def __getitem__(self, idx):
        return {
                "z": self.records[idx]["z"][self.pitch_range[0]:self.pitch_range[1]],
                "text": self.compose_prompt(self.records[idx])
                }
    
    def check_for_nans(self):
        for r in self.records:
            # check if z has nan values
            if np.isnan(r["z"]).any():
                raise ValueError("Nan values in z")
            
    def get_z_shape(self):
        shapes = [r["z"].shape for r in self.records]
        # return unique shapes
        return list(set(shapes))

    
if __name__ == "__main__":

    # set seed
    SEED = 0
    torch.manual_seed(SEED)

    BATCH_SIZE = 1
    LATENT_T = 86

    # initialize wandb logger
    wandb.init()
    logger = WandbLogger(project="synth_flow")

    # don't log models
    wandb.config.log_model = False

    DATASET = "dataset_a"
    if DATASET == "dataset_a":
        PITCH_RANGE = [2,12]

        trn_ds = EncodedAudioDataset([f"artefacts/synth_data_{i}.pt" for i in range(9)], PITCH_RANGE)
        trn_ds.check_for_nans()
        trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)

        val_ds = EncodedAudioDataset([f"artefacts/synth_data_9.pt"], PITCH_RANGE)
        val_ds.check_for_nans()
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)

        
    elif DATASET == "dataset_b":

        PITCH_RANGE = [0,10]
        trn_ds = EncodedAudioDataset([f"artefacts/synth_data_2_joined_{i}.pt" for i in range(3)], PITCH_RANGE)
        trn_ds.check_for_nans()
        trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True)

        val_ds = EncodedAudioDataset([f"artefacts/synth_data_2_joined_3.pt"], PITCH_RANGE)
        val_ds.check_for_nans()
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    src_model = get_pretrained_model("stabilityai/stable-audio-open-1.0")[0].to("cpu")
    src_model = src_model.to("cpu")
    transformer_model = src_model.model.model
    transformer_model = transformer_model.train()
    text_conditioner = src_model.conditioner.conditioners.prompt

    t5_version = "google-t5/t5-base"

    
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='step')

    model = FlowMatchingModule(
    main_model=transformer_model,
    text_conditioner=text_conditioner,
    n_channels=PITCH_RANGE[1] - PITCH_RANGE[0],
    t_input=LATENT_T,
    )

    trainer = pl.Trainer(devices = [3], logger=logger, gradient_clip_val=1.0, callbacks=[lr_callback], max_epochs=1000, precision="16-mixed")

    trainer.fit(model, trn_dl, val_dl, ckpt_path="synth_flow/9gzpz0i6/epoch=85-step=774000.ckpt")
    # save checkpoint
    trainer.save_checkpoint("artefacts/model_finetuned_2.ckpt")

    

        
