import optuna
from utils import read_optim_config, set_seeds
from utils import create_folder_if_not_exists
from utils import parse_sampling_strategy
from data_process3d import DataProcess3d
from train import Trainer
import argparse
from datetime import datetime
import os
from utils import RunStudy
import yaml

def save_study_results(study, result_fold, config):
    """Save study results locally in YAML format"""
    # Create results directory if it doesn't exist
    study_results_dir = os.path.join(result_fold, 'optuna_results')
    os.makedirs(study_results_dir, exist_ok=True)
    
    # Save best trial information
    if len(config['optuna_loss']) > 1:
        # Multi-objective optimization
        best_trials = study.best_trials
        best_results = []
        for i, trial in enumerate(best_trials):
            best_results.append({
                'number': trial.number,
                'params': trial.params,
                'values': trial.values,
                'direction': config['direction'][i] if i < len(config['direction']) else 'minimize'
            })
        
        with open(os.path.join(study_results_dir, 'best_trials.yaml'), 'w') as f:
            yaml.dump(best_results, f, default_flow_style=False, sort_keys=False)
    else:
        # Single-objective optimization
        best_result = {
            'number': study.best_trial.number,
            'value': study.best_value,
            'params': study.best_params
        }
        with open(os.path.join(study_results_dir, 'best_trial.yaml'), 'w') as f:
            yaml.dump(best_result, f, default_flow_style=False, sort_keys=False)
    
    # Save all trials
    all_trials = []
    for trial in study.trials:
        trial_info = {
            'number': trial.number,
            'state': str(trial.state),
            'params': trial.params,
            'values': trial.values if trial.values else None
        }
        all_trials.append(trial_info)
    
    with open(os.path.join(study_results_dir, 'all_trials.yaml'), 'w') as f:
        yaml.dump(all_trials, f, default_flow_style=False, sort_keys=False)
    
    # Save study statistics
    stats = {
        'study_name': config['sweep_name'],
        'n_trials': len(study.trials),
        'n_complete': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'datetime': datetime.now().isoformat(),
        'objectives': config['optuna_loss'],
        'directions': config['direction'],
        'sampling_strategy': config['sampling_strategy'],
        'config': config
    }
    
    with open(os.path.join(study_results_dir, 'study_stats.yaml'), 'w') as f:
        yaml.dump(stats, f, default_flow_style=False, sort_keys=False)
    
    # Create a simple text summary (keeping this as text for readability)
    with open(os.path.join(study_results_dir, 'summary.txt'), 'w') as f:
        f.write(f"Optuna Study Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Study name: {config['sweep_name']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of trials: {len(study.trials)}\n")
        f.write(f"Number of objectives: {len(config['optuna_loss'])}\n")
        f.write(f"Directions: {config['direction']}\n\n")
        
        if len(config['optuna_loss']) > 1:
            f.write("Best Trials:\n")
            for i, trial in enumerate(study.best_trials):
                f.write(f"  Trial {trial.number}:\n")
                f.write(f"    Values: {trial.values}\n")
                f.write(f"    Parameters:\n")
                for param_name, param_value in trial.params.items():
                    f.write(f"      {param_name}: {param_value}\n")
        else:
            f.write(f"Best Trial: {study.best_trial.number}\n")
            f.write(f"Best Value: {study.best_value}\n")
            f.write(f"Best Parameters:\n")
            for param_name, param_value in study.best_params.items():
                f.write(f"  {param_name}: {param_value}\n")

class LocalProgressCallback:
    """Simple callback for local progress tracking"""
    def __init__(self, result_fold):
        self.result_fold = result_fold
        self.trial_results = []
        
    def __call__(self, study, trial):
        print(f"\n--- Trial {trial.number} completed ---")
        print(f"State: {trial.state}")
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(f"Values: {trial.values}")
            print(f"Parameters: {trial.params}")
            
            # Save individual trial result
            trial_result = {
                'number': trial.number,
                'values': trial.values,
                'params': trial.params,
                'datetime': datetime.now().isoformat()
            }
            self.trial_results.append(trial_result)
            
            # Save intermediate results in YAML
            results_dir = os.path.join(self.result_fold, 'optuna_results')
            os.makedirs(results_dir, exist_ok=True)
            
            with open(os.path.join(results_dir, 'trial_results.yaml'), 'w') as f:
                yaml.dump(self.trial_results, f, default_flow_style=False, sort_keys=False)
            
            # Also save individual trial file for easy access
            trial_file = os.path.join(results_dir, f'trial_{trial.number:04d}.yaml')
            with open(trial_file, 'w') as f:
                yaml.dump(trial_result, f, default_flow_style=False, sort_keys=False)

def init_sweep(config, dataset, result_fold, attr_config=None):
    if len(config['optuna_loss']) > 1:
        print(f'Multi criterion hyperparam optimization is picked! {config["optuna_loss"]}')
    else:
        print(f'Single-target hyperparam optimization is picked! {config["optuna_loss"][0]}')    
    
    opt_direction = config['direction']
    if dataset is not None:
        def objective(trial):
            run_study = RunStudy(config, result_fold, trial) 
            trainer = Trainer(config, dataset, result_fold, opt_direction,
                            run_study=run_study, trial=trial)  # Note: run_study is now None
            return trainer.train(trial)
    else:
        def objective(trial):
            run_study = RunStudy(config, result_fold, trial)
            attr_num = trial.suggest_categorical('attr_ind', 
                                                  list(attr_config.keys()))
            config['data_paths']['attrib_names'] = attr_config[attr_num].copy()
            dataset = DataProcess3d(config, trial)
            trainer = Trainer(config, dataset, result_fold, opt_direction,
                            run_study=run_study, trial=trial)  # Note: run_study is now None
            return trainer.train(trial)
    
    print(f'Optuna targets {config["optuna_loss"]} has been taken')
    
    # Create sampler
    sampler = parse_sampling_strategy(sampling_strategy=config['sampling_strategy'])
    
    # Create study
    if len(config['optuna_loss']) > 1:
        study = optuna.create_study(directions=opt_direction, 
                                   sampler=sampler,
                                   study_name=config['sweep_name'],
                                   storage=None,  # Use in-memory storage
                                   load_if_exists=False)
    else:
        study = optuna.create_study(direction=opt_direction[0], 
                                   sampler=sampler,
                                   study_name=config['sweep_name'],
                                   storage=None,
                                   load_if_exists=False)
    
    # Create local callback for progress tracking
    local_callback = LocalProgressCallback(result_fold)
    
    # Save study configuration
    study_results_dir = os.path.join(result_fold, 'optuna_results')
    os.makedirs(study_results_dir, exist_ok=True)
    
    # Save the configuration used for this study
    # with open(os.path.join(study_results_dir, 'study_config.yaml'), 'w') as f:
    #     yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    # Run optimization
    study.optimize(objective, 
                  n_trials=config['sweep_num'], 
                  callbacks=[local_callback],
                  show_progress_bar=True)  # This will show a progress bar
    
    # Save final results
    save_study_results(study, result_fold, config)
    
    # Print final results
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETED")
    print("="*50)
    
    if len(config['optuna_loss']) > 1:
        print("\nBest trials:")
        for i, trial in enumerate(study.best_trials):
            print(f"  Trial {trial.number}: {trial.values}")
            print(f"    Params: {trial.params}")
    else:
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best value: {study.best_value}")
        print(f"Best params: {study.best_params}")
    
    print(f"\nResults saved in: {study_results_dir}")

def main(config):
    set_seeds(config['seed'])
    result_fold = create_folder_if_not_exists(config)
    if config['fixed_attrs']:
        dataset = DataProcess3d(config)
        init_sweep(config=config, dataset=dataset, result_fold=result_fold, attr_config=None)
    else:
        attr_conf_path = config['diff_attrs_path']
        attr_config = read_optim_config(attr_conf_path)
        init_sweep(config=config, dataset=None, result_fold=result_fold, attr_config=attr_config)
    print('Sweep completed!')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/ae3d.yaml')
    args = parser.parse_args()

    config = read_optim_config(args.config)
    main(config)