import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class Conv3DEncoder(nn.Module):
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],  # (c, d, h, w)
        latent_dim: int = 32,
        base_filters: int = 32,
        num_conv_layers: int = 4,
        kernel_size: tuple = (3,3,1),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0,
        use_avgpool: bool = True  
    ):
    
        super(Conv3DEncoder, self).__init__()
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.num_conv_layers = num_conv_layers
        self.use_batch_norm = use_batch_norm
        self.use_avgpool = use_avgpool
        
        channels, depth, height, width = input_shape
        
        self.conv_blocks = nn.ModuleList()
        current_channels = channels
        current_depth, current_height, current_width = depth, height, width
        
        for i in range(num_conv_layers):
            out_channels = base_filters * (2 ** i)
            
            stride_d = 2 if current_depth > 4 else 1
            stride_h = 2 if current_height > 4 else 1
            stride_w = 2 if current_width > 4 else 1
            
            pad_d = kernel_size[0] // 2
            pad_h = kernel_size[1] // 2
            pad_w = kernel_size[2] // 2
            
            layers = [
                nn.Conv3d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(stride_d, stride_h, stride_w),
                    padding=(pad_d, pad_h, pad_w),
                    bias=not use_batch_norm
                )
            ]
            
            if use_batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            
            layers.append(nn.ReLU(inplace=True))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout3d(dropout_rate))
            
            self.conv_blocks.append(nn.Sequential(*layers))
            
            current_channels = out_channels
            current_depth = self._conv_output_size(current_depth, kernel_size[0], stride_d, pad_d)
            current_height = self._conv_output_size(current_height, kernel_size[1], stride_h, pad_h)
            current_width = self._conv_output_size(current_width, kernel_size[2], stride_w, pad_w)
        
        if use_avgpool:
            self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.flattened_size = current_channels 
        else:
            self.adaptive_pool = None
            self.flattened_size = current_channels * current_depth * current_height * current_width
        
        print(f"Encoder flattened size: {self.flattened_size}")
        
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, latent_dim)
        
        self._initialize_weights()
    
    def _conv_output_size(self, input_size, kernel_size, stride, padding):
        return (input_size + 2 * padding - kernel_size) // stride + 1
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        if self.adaptive_pool is not None:
            x = self.adaptive_pool(x)
        
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class Conv3DDecoder(nn.Module):
    
    def __init__(
        self,
        output_shape: Tuple[int, int, int, int], 
        latent_dim: int = 32,
        base_filters: int = 32,
        num_conv_layers: int = 4,
        kernel_size: tuple = (3,3,1),
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Args:
            output_shape: Target output shape (channels, depth, height, width)
            latent_dim: Size of latent vector
            base_filters: Number of filters in last convolutional layer (will be scaled down)
            num_conv_layers: Number of transposed convolutional layers
            kernel_size: Size of convolutional kernels
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0 = no dropout)
        """
        super(Conv3DDecoder, self).__init__()
        
        self.output_shape = output_shape
        self.latent_dim = latent_dim
        self.base_filters = base_filters
        self.num_conv_layers = num_conv_layers
        self.use_batch_norm = use_batch_norm
        
        channels, depth, height, width = output_shape
        
        # We need to calculate the spatial dimensions before the transposed conv layers
        # Starting from the smallest dimensions and progressively upsample
        current_depth, current_height, current_width = depth, height, width
        
        # Calculate dimensions at each layer (working backwards)
        decoder_dims = []
        for i in range(num_conv_layers):
            decoder_dims.append((current_depth, current_height, current_width))
            
            # Determine upsampling factors (same logic as encoder but reversed)
            stride_d = 2 if current_depth > 4 else 1
            stride_h = 2 if current_height > 4 else 1
            stride_w = 2 if current_width > 4 else 1
            
            # Calculate input size for next layer (previous in forward pass)
            current_depth = self._conv_transpose_input_size(current_depth, kernel_size[0], stride_d)
            current_height = self._conv_transpose_input_size(current_height, kernel_size[1], stride_h)
            current_width = self._conv_transpose_input_size(current_width, kernel_size[2], stride_w)
        
        # Reverse to get dimensions in forward order
        decoder_dims = decoder_dims[::-1]
        
        # Initial linear layer to project latent vector to appropriate size
        self.initial_depth, self.initial_height, self.initial_width = (current_depth, 
                                                                        current_height, 
                                                                        current_width)
        initial_channels = base_filters * (2 ** (num_conv_layers - 1))
        self.initial_linear_size = initial_channels * current_depth * current_height * current_width
        
        print(f"Decoder initial linear size: {self.initial_linear_size}")
        print(f"Decoder initial spatial dims: ({current_depth}, {current_height}, {current_width})")
        
        self.fc = nn.Linear(latent_dim, self.initial_linear_size)
        
        # Build transposed convolutional blocks
        self.deconv_blocks = nn.ModuleList()
        
        for i in range(num_conv_layers):
            in_channels = base_filters * (2 ** (num_conv_layers - 1 - i))
            out_channels = base_filters * (2 ** (num_conv_layers - 2 - i)) if i < num_conv_layers - 1 else channels
            
            # Get target dimensions for this layer
            target_depth, target_height, target_width = decoder_dims[i]
            
            # Calculate stride and padding for transposed conv
            stride_d = 2 if target_depth > current_depth else 1
            stride_h = 2 if target_height > current_height else 1
            stride_w = 2 if target_width > current_width else 1
            
            # Calculate padding
            pad_d = kernel_size[0] // 2
            pad_h = kernel_size[1] // 2
            pad_w = kernel_size[2] // 2
            
            out_pad_d_calc = target_depth - self._conv_transpose_output_size(current_depth, 
                                                                            kernel_size[0], 
                                                                            stride_d, pad_d)
            out_pad_h_calc = target_height - self._conv_transpose_output_size(current_height, 
                                                                             kernel_size[1], 
                                                                             stride_h, pad_h)
            out_pad_w_calc = target_width - self._conv_transpose_output_size(current_width, 
                                                                            kernel_size[2], 
                                                                            stride_w, pad_w)
            
            # Constrain output padding to be less than stride
            out_pad_d = min(max(0, out_pad_d_calc), stride_d - 1)
            out_pad_h = min(max(0, out_pad_h_calc), stride_h - 1)
            out_pad_w = min(max(0, out_pad_w_calc), stride_w - 1)
            
            # Build transposed convolutional block
            layers = [
                nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=(stride_d, stride_h, stride_w),
                    padding=(pad_d, pad_h, pad_w),
                    output_padding=(out_pad_d, out_pad_h, out_pad_w),
                    bias=not use_batch_norm
                )
            ]
            
            if use_batch_norm and i < num_conv_layers - 1:  # No batch norm on last layer
                layers.append(nn.BatchNorm3d(out_channels))
            
            if i < num_conv_layers - 1:  # No activation on last layer
                layers.append(nn.ReLU(inplace=True))
            
            if dropout_rate > 0 and i < num_conv_layers - 1:
                layers.append(nn.Dropout3d(dropout_rate))
            
            self.deconv_blocks.append(nn.Sequential(*layers))
            
            # Update current dimensions
            current_depth, current_height, current_width = target_depth, target_height, target_width
        
        self._initialize_weights()
    
    def _conv_transpose_input_size(self, output_size, kernel_size, stride):
        """Calculate input size needed for transposed conv to achieve given output size"""
        return (output_size - kernel_size + 2 * (kernel_size // 2)) // stride + 1
    
    def _conv_transpose_output_size(self, input_size, kernel_size, stride, padding):
        """Calculate output size of transposed convolution"""
        return (input_size - 1) * stride - 2 * padding + kernel_size
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        
        # Project and reshape
        x = self.fc(z)
        x = x.view(x.size(0), -1, self.initial_depth, 
                    self.initial_height, 
                    self.initial_width)
        
        # Apply transposed convolutional blocks
        for deconv_block in self.deconv_blocks:
            x = deconv_block(x)
        
        return x


class Conv3DVAE(nn.Module):
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        latent_dim: int = 32,
        base_filters: int = 32,
        enc_conv_layers = 2,
        dec_conv_layers = 2,
        enc_dropout = 0.05,
        dec_dropout = 0.1,
        use_pool_enc=True,
        kernel_size: int = (3, 3, 1),
        min_bits=0.01,
        use_batch_norm: bool = True,
        beta: float = 1.0  
    ):
        
        super(Conv3DVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.base_filters = base_filters
        self.enc_conv_layers = enc_conv_layers
        self.dec_conv_layers = dec_conv_layers
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.kernel_size = kernel_size
        self.min_bits = min_bits

        self.encoder = Conv3DEncoder(
            input_shape=input_shape,
            latent_dim=latent_dim,
            base_filters=base_filters,
            num_conv_layers=enc_conv_layers,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=enc_dropout,
            use_avgpool=use_pool_enc
        )
        
        self.decoder = Conv3DDecoder(
            output_shape=input_shape,
            latent_dim=latent_dim,
            base_filters=base_filters,
            num_conv_layers=dec_conv_layers,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dec_dropout
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, x, categor_attrs=None):
        
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, reduction: str = 'mean'):
        
        recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
        if self.training:
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) 
            kl_per_dim = kl_per_dim.mean(0) 
            min_bits = torch.tensor(self.min_bits).to(kl_per_dim.device)
            kl_per_dim = torch.max(kl_per_dim, min_bits) 
            kl_div = kl_per_dim.mean()
        else:
            kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_div = kl_div.mean()
        
        total_loss = recon_loss + kl_div
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_div
            }
    
    def encode(self, x, categorical_x=None):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_latent(self, x):
        mu, logvar = self.encoder(x)
        return self.reparameterize(mu, logvar)


class Conv3DVAE_Cat(nn.Module):
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        n_categories: int,  # Number of unique categories
        embedding_dim: int = 8,  # Size of embedding for each category
        latent_dim: int = 32,
        base_filters: int = 32,
        enc_conv_layers = 2,
        dec_conv_layers = 2,
        enc_dropout = 0.05,
        dec_dropout = 0.1,
        use_pool_enc=True,
        kernel_size: int = (3, 3, 1),
        min_bits=0.01,
        use_batch_norm: bool = True,
        beta: float = 1.0  
    ):
        
        super(Conv3DVAE, self).__init__()
        
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.beta = beta
        self.base_filters = base_filters
        self.enc_conv_layers = enc_conv_layers
        self.dec_conv_layers = dec_conv_layers
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.kernel_size = kernel_size
        self.min_bits = min_bits
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        
        channels, depth, height, width = input_shape
        
        # Embedding layer for categorical feature
        # Assumes categorical values are integers from 0 to n_categories-1
        self.category_embedding = nn.Embedding(n_categories, embedding_dim)
        
        # Modified input shape: original channels + embedding_dim
        modified_input_shape = (channels + embedding_dim, depth, height, width)

        self.encoder = Conv3DEncoder(
            input_shape=modified_input_shape,
            latent_dim=latent_dim,
            base_filters=base_filters,
            num_conv_layers=enc_conv_layers,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=enc_dropout,
            use_avgpool=use_pool_enc
        )
        
        self.decoder = Conv3DDecoder(
            output_shape=modified_input_shape,  # Decoder outputs combined features
            latent_dim=latent_dim,
            base_filters=base_filters,
            num_conv_layers=dec_conv_layers,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout_rate=dec_dropout
        )
        
        # Final layer to project back to original channels (remove embedding dim)
        self.channel_reduction = nn.Conv3d(
            in_channels=channels + embedding_dim,
            out_channels=channels,
            kernel_size=1
        )
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def forward(self, continuous_x, categorical_x):
        """
        Args:
            continuous_x: Continuous attributes, shape (batch, channels, depth, height, width)
            categorical_x: Categorical cube, shape (batch, 1, depth, height, width) 
                          with integer values
        """
        # Convert categorical to embeddings
        # categorical_x: (batch, 1, d, h, w) -> (batch, d, h, w)
        categorical_x = categorical_x.squeeze(1).long()
        
        # Get embeddings for each spatial position
        # Output: (batch, d, h, w, embedding_dim)
        embedded = self.category_embedding(categorical_x)
        
        # Reshape to match convolution format
        # (batch, d, h, w, embedding_dim) -> (batch, embedding_dim, d, h, w)
        embedded = embedded.permute(0, 4, 1, 2, 3)
        
        # Concatenate along channel dimension
        combined_x = torch.cat([continuous_x, embedded], dim=1)
        
        # Encode
        mu, logvar = self.encoder(combined_x)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon_combined = self.decoder(z)
        
        # Reduce channels back to original
        recon_x = self.channel_reduction(recon_combined)
        
        return recon_x, mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar, reduction: str = 'mean'):
        # x here is the continuous attributes only (for loss calculation)
        recon_loss = F.mse_loss(recon_x, x, reduction=reduction)
        if self.training:
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) 
            kl_per_dim = kl_per_dim.mean(0) 
            min_bits = torch.tensor(self.min_bits).to(kl_per_dim.device)
            kl_per_dim = torch.max(kl_per_dim, min_bits) 
            kl_div = kl_per_dim.mean()
        else:
            kl_div = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            kl_div = kl_div.mean()
        
        total_loss = recon_loss + kl_div
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_div
            }
    
    def encode(self, continuous_x, categorical_x):
        """Encode to latent distribution parameters"""
        categorical_x = categorical_x.squeeze(1).long()
        embedded = self.category_embedding(categorical_x)
        embedded = embedded.permute(0, 4, 1, 2, 3)
        combined_x = torch.cat([continuous_x, embedded], dim=1)
        return self.encoder(combined_x)
    
    def decode(self, z):
        """Decode latent vector to reconstruction (continuous part only)"""
        recon_combined = self.decoder(z)
        return self.channel_reduction(recon_combined)
    
    def get_latent(self, continuous_x, categorical_x):
        """Get latent representation"""
        mu, logvar = self.encode(continuous_x, categorical_x)
        return self.reparameterize(mu, logvar)


def create_seismic_vae(dxy,
                       dt, 
                       config, trial,
                        **kwargs
) -> Conv3DVAE:
    
    dt = dt
    dx = dy = dxy
    # dy = config['dxy']
    attrs = config['data_paths']['attrib_names']
    n_attributes = len(attrs)
    base_filters = config['base_filts']
    input_shape = (n_attributes, dx, dy, dt)
    dropout_rate = config['dropout']
    # latent_dim = config['laten_vec_size']
    dec_conv_layers = config['dec_conv_layers']
    enc_conv_layers = config['enc_conv_layers']
    use_pool_enc = config['use_pool_enc']
    
    if 'base_filts' in config['parameters'].keys():
        base_filts = config['parameters']['base_filts']
        base_filts = trial.suggest_categorical(base_filts['name'],
                                                    base_filts['values'])
    else:
        base_filts = enc_conv_layers

    #-------------------------------------------------------------------

    if 'enc_conv_layers' in config['parameters'].keys():
        enc_conv = config['parameters']['enc_conv_layers']
        enc_conv_layers = trial.suggest_categorical(enc_conv['name'],
                                                    enc_conv['values'])
    else:
        enc_conv_layers = enc_conv_layers
    #-------------------------------------------------------------------
    if 'dec_conv_layers' in config['parameters'].keys():
        dec_conv = config['parameters']['dec_conv_layers']
        dec_conv_layers = trial.suggest_categorical(dec_conv['name'],
                                                    dec_conv['values'])
    else:
        dec_conv_layers = dec_conv_layers 
    #-------------------------------------------------------------------
    if 'dec_dropout' in config['parameters'].keys():
        dec_dropout = config['parameters']['dec_dropout']
        dec_dropout = trial.suggest_float(dec_dropout['name'],
                                      low=float(dec_dropout['low']),
                                      high=float(dec_dropout['high']),
                                      log=True)
    elif 'dec_dropout' in config.keys():
        dec_dropout = config['dec_dropout']
    else:
        dec_dropout = dropout_rate 
    
    enc_dropout = dropout_rate
    #-----------------------------------------------------------------------
    if 'latent_dim' in config['parameters'].keys():
        latent_dim = config['parameters']['latent_dim']
        latent_dim = trial.suggest_categorical(latent_dim['name'],
                                               latent_dim['values'])
    else:
         latent_dim = config['latent_dim']


    print('Encoder:')
    print(f'Dropout_rate: {enc_dropout}, conv_layers: {enc_conv_layers}, base_filts: {base_filters}, use_avg_pool: {use_pool_enc}\n')
    print('Decoder:')
    print(f'Dropout_rate: {dec_dropout}, conv_layers: {dec_conv_layers}, base_filts: {base_filters}')
    
    use_categoric = sum(['categoric' in x for x in attrs])

    if not use_categoric:
        print('Take VAE only for categorical data!')
        model = Conv3DVAE(
            input_shape=input_shape,
            latent_dim=latent_dim,
            base_filters=base_filters,
            enc_conv_layers=enc_conv_layers,
            dec_conv_layers=dec_conv_layers,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            use_pool_enc=use_pool_enc
            )
    else:
        print('Take VAE for both categorical and continous data!')
        min_mask_ratio = config['min_mask_ratio']
        categories_num = (config['big_facies_num'] + 1 
                          if min_mask_ratio < 1 else config['big_facies_num'])
        embedding_dim = min(6, latent_dim // 2)

        model = Conv3DVAE_Cat(
            input_shape=input_shape,
            n_categories=categories_num,  # Number of unique categories
            embedding_dim=embedding_dim,
            latent_dim=latent_dim,
            base_filters=base_filters,
            enc_conv_layers=enc_conv_layers,
            dec_conv_layers=dec_conv_layers,
            enc_dropout=enc_dropout,
            dec_dropout=dec_dropout,
            use_pool_enc=use_pool_enc
            )

    # input_shape: Tuple[int, int, int, int],
        # n_categories: int,  # Number of unique categories
        # embedding_dim: int = 8,  # Size of embedding for each category
        # latent_dim: int = 32,
        # base_filters: int = 32,
        # enc_conv_layers = 2,
        # dec_conv_layers = 2,
        # enc_dropout = 0.05,
        # dec_dropout = 0.1,
        # use_pool_enc=True,
        # kernel_size: int = (3, 3, 1),
        # min_bits=0.01,
        # use_batch_norm: bool = True,
        # beta: float = 1.0 
    print(f"Created 3D VAE for input shape: {input_shape}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

