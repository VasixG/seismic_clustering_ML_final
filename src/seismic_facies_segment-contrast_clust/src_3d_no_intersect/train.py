from models import create_seismic_vae
import torch
import yaml
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import optuna
from inference import Inference
from utils import RunStudy

class Trainer:

    def __init__(self, config,
                 data_cl_inst,
                 result_fold,
                 opt_direction, run_study,
                 trial):

        self.run_study = run_study
        self.config = config
        self.opt_direction = opt_direction
        self.es_tolerance = config['es_tolerance']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Device: {self.device}')
        self.patch_size = self._get_patch_size(trial)
        
        dxy, dt = self.patch_size[0], self.patch_size[-1]
        self.model = create_seismic_vae(dxy, 
                                        dt, config, trial)
        
        self.model.to(self.device)
        self.best_state_dct = self.model.state_dict()
        self.data_cl_inst = data_cl_inst
        if 'warmup_epochs' in config['parameters'].keys():
            
            warm_ep = config['parameters']['warmup_epochs']
            self.warmup_epochs = trial.suggest_categorical(warm_ep['name'],
                                                           warm_ep['values'])
            print(f'Took warmup epchs for B: {self.warmup_epochs} in the run.')
        else:
            self.warmup_epochs = config['warmup_epochs']
            print(f'Warmup epchs for B is constant: {self.warmup_epochs} for all runs.')
        # self.bs = trial.suggest_categorical(config['parameters']['batch_size']['name'],
        #                                     config['parameters']['batch_size']['values'])
        self.train_bs = config['train_bs']

        self.train_stat = self.data_cl_inst.cube_stats

        self.train_loader, self.test_loader = self.data_cl_inst.create_seismic_dataloaders(dxy, 
                                                                                           dt,
                                                                                           training=True)
        print(f'len train dataset {len(self.train_loader.dataset)}')
        self.len_train = len(self.train_loader.dataset)
        self.len_test = len(self.test_loader.dataset)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          lr=float(config['lr']),
                                          weight_decay=config['w_decay'])

        self.result_fold = result_fold
        # self.l_m = Loss_Metrics_Mem(current_target, result_fold)
        # self.run_study = RunStudy(config, result_fold, trial)
        self.save_model_dict(trial)
        self.sched_patience = self.es_tolerance // 2
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=self.sched_patience,
                                           min_lr=1e-5, verbose=True)
        self.n_epochs = config['epochs']
        
        self.separate_metrics = {}      

    def train(self, trial):
        """
        Train the VAE model with logging to Neptune
        
        Args:
            trial: Optuna trial object for hyperparameter optimization
        """
        print(f"Starting training for trial {trial.number}")
        
        # Initialize best loss tracking
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.n_epochs):
            # Training phase
            self.model.train()
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            train_total_loss = 0.0
            
            for batch_data in self.train_loader:
                # print(f'batch_data type: {type(batch_data)}') if i==0 else None
                attrs, categor_attrs = batch_data
                attrs = attrs.to(self.device)
                categor_attr = categor_attrs.to(self.device) if categor_attrs is not None else None
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_attrs, mu, logvar, z = self.model(attrs, categor_attr)
                
                # Calculate loss
                loss_dict = self.model.loss_function(recon_attrs, 
                                                    attrs, 
                                                    mu, logvar)
                
                # Backward pass
                beta = min(1.0, epoch / self.warmup_epochs)  # Linear warmup
                loss = loss_dict['recon_loss'] + beta * loss_dict['kl_loss']
                loss.backward()
                # loss_dict['loss'].backward()
                
                # Gradient clipping (optional, helps with stability)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update weights
                self.optimizer.step()
                
                # Accumulate losses
                train_total_loss += loss_dict['loss'].item()
                train_recon_loss += loss_dict['recon_loss'].item()
                train_kl_loss += loss_dict['kl_loss'].item()
            
            # Calculate average training losses
            avg_train_total = train_total_loss / len(self.train_loader)
            avg_train_recon = train_recon_loss / len(self.train_loader)
            avg_train_kl = train_kl_loss / len(self.train_loader)
            
            # Validation phase
            self.model.eval()
            val_recon_loss = 0.0
            val_kl_loss = 0.0
            val_total_loss = 0.0
            
            with torch.no_grad():
                for batch_data in self.test_loader:
                    attrs, categor_attrs = batch_data
                    attrs = attrs.to(self.device)
                    categor_attrs = categor_attrs.to(self.device) if categor_attrs is not None else None
                    # Forward pass
                    recon_attrs, mu, logvar, z = self.model(attrs, categor_attrs)
                    
                    # Calculate loss
                    loss_dict = self.model.loss_function(recon_attrs, attrs, mu, logvar)
                    
                    # Accumulate losses
                    val_total_loss += loss_dict['loss'].item()
                    val_recon_loss += loss_dict['recon_loss'].item()
                    val_kl_loss += loss_dict['kl_loss'].item()
            
            # Calculate average validation losses
            avg_val_total = val_total_loss / len(self.test_loader)
            avg_val_recon = val_recon_loss / len(self.test_loader)
            avg_val_kl = val_kl_loss / len(self.test_loader)
            
            # Log to Neptune
            # Training losses
            self.run_study.append(f"train_errors/total_loss_{trial.number}.pdf", avg_train_total)
            self.run_study.append(f"train_errors/recon_loss_{trial.number}.pdf", avg_train_recon)
            self.run_study.append(f"train_errors/kl_loss_{trial.number}.pdf", avg_train_kl)
            
            # Validation losses
            self.run_study.append(f"test_errors/total_loss_{trial.number}.pdf", avg_val_total)
            self.run_study.append(f"test_errors/recon_loss_{trial.number}.pdf", avg_val_recon)
            # self.run_study[f"test_errors/kl_loss_{trial.number}"].append(avg_val_kl)
            self.run_study.append(f"test_errors/kl_loss_{trial.number}.pdf", avg_val_kl)
            
            # Update learning rate scheduler based on validation loss
            self.scheduler.step(avg_val_total)
            
            if self._check_nans_in_weights():
                print('Nans in model during training! Stopping early.')
                if self.opt_direction == 'maximize':
                    return -1e10
                else:
                    return 1e10
            # Print progress
            if epoch % 5 == 0:
                print(f"Trial {trial.number}, Epoch {epoch}/{self.n_epochs}")
                print(f"  Train - Total: {avg_train_total:.4f}, Recon: {avg_train_recon:.4f}, KL: {avg_train_kl:.4f}")
                print(f"  Val   - Total: {avg_val_total:.4f}, Recon: {avg_val_recon:.4f}, KL: {avg_val_kl:.4f}")
            
            # Check for best model
            if round(avg_val_total, 3) < round(best_val_loss, 3):
                best_val_loss = avg_val_total
                self.best_state_dct = self.model.state_dict().copy()
                patience_counter = 0
                
                # Optionally save best model checkpoint

                torch.save(self.best_state_dct, 
                           f"{self.result_fold}/weights/best_model_{trial.number}.pt")
            else:
                patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.es_tolerance:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Report best validation loss to Optuna
        trial.report(best_val_loss, step=self.n_epochs)
        
        # Log best validation loss to Neptune
        # self.run_study[f"best_validation/total_loss_{trial.number}"].append(best_val_loss)
        
        # Load best model state
        self.model.load_state_dict(self.best_state_dct)
        self.run_study.save_plots()
        
        #TODO inference here
        ar_score = self.make_inference(trial)
        print(f"Trial {trial.number} completed. Best validation loss: {best_val_loss:.4f}")
        return ar_score

    def _get_patch_size(self, trial):
        
        if 'dxy' not in self.config['parameters'].keys():
            dxy = self.config['dxy']
        else: 
            params = self.config['parameters']
            dxy = trial.suggest_categorical(params['dxy']['name'],
                                        params['dxy']['values'])
            # dt = trial.suggest_categorical(params['dt']['name'],
            #                             params['dt']['values'])
            # patch_size = (dxy, dxy, dt)
        if 'dt' not in self.config['parameters'].keys():
            dt = self.config['dt']
        else: 
            params = self.config['parameters']
            dt = trial.suggest_categorical(params['dt']['name'],
                                        params['dt']['values'])
        
        patch_size = (dxy, dxy, dt)
        return patch_size

    @torch.no_grad()
    def _check_nans_in_weights(self):
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                return True
        return False

    def save_model_dict(self, trial):
        model_dict = {}
        model_dict['input_shape'] = self.model.input_shape
        model_dict['latent_dim'] = self.model.latent_dim
        model_dict['base_filters'] = self.model.base_filters
        model_dict['enc_conv_layers'] = self.model.enc_conv_layers
        model_dict['dec_conv_layers'] = self.model.dec_conv_layers
        model_dict['enc_dropout'] = self.model.enc_dropout
        model_dict['dec_dropout'] = self.model.dec_dropout
        model_dict['kernel_size'] = self.model.kernel_size
        # model_dict['use_batch_norm'] = self.model.use_batch_norm
        yaml_path = os.path.join(self.result_fold, 'weights', 
                                 f'model_config_{trial.number}.yaml')
        with open(yaml_path, 'w') as file:
            yaml.dump(model_dict, file)
        return model_dict

    def make_inference(self, trial):
        # Code conditional inference
        self.model.eval()

        infer_inst = Inference(self.patch_size,
                                self.data_cl_inst,
                                self.result_fold,
                                self.config,
                                self.model,
                                self.device,
                                self.run_study,
                                trial
                                 )
        trail_score = infer_inst.run_complete_inference()
        return trail_score

