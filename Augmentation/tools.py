from pathlib import Path
import os, sys
import numpy as np
import json
from tqdm import tqdm
from copy import deepcopy

import matplotlib.pyplot as plt
#import plotly.express as px
#import plotly.graph_objs as go

import torch
from kan import KAN
#from kan.MLP import MLP
from kan.LBFGS import LBFGS
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split

def root_mean_squared_error(y_true, y_pred, kwargs={}):
    return (mean_squared_error(y_true, y_pred, **kwargs))**0.5


def JSON_Create(diction: dict, FileDirectory: str, FileName: str) -> None:
    """
    Function for creating JSON log-file with dictionary.

    :param diction: Dictionary for writing
    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file. Should be ended with ".txt"
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    os.makedirs(FileDirectory, exist_ok=True)  # Creating / checking existing of file-path
    with open(filename, 'w') as f:
        json.dump(diction, f, indent=4)  # Writing file


def JSON_Read(FileDirectory: str, FileName: str) -> dict:
    """
    Function for loading dictionary from log-file.

    :param FileDirectory: Path to logging file
    :param FileName: Name of logging file
    """
    filename = Path(FileDirectory) / FileName  # Full file-path with file-name
    with open(filename) as f:
        return json.load(f)  # Loading dictionary


class KAN_es(KAN):
    """
    KAN class with early stopping training. Early sropping was made closly to skl.MLPRegressor.
    Implemented fit() predict() methods for using KAN_es in scikit-learn`s functions.
    """
    '''
    def __init__(self, *args, validation_fraction = 0.1,train_kwargs={}, **kwargs):
        super(KAN_es, self).__init__(*args, **kwargs)
        self.validation_fraction = validation_fraction
        self.train_kwargs = train_kwargs
    '''
        
    def train_es(self, dataset, tol=0.001, n_iter_no_change=10,
                  opt="LBFGS", steps=100, log=1, lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., update_grid=True, grid_update_num=10, loss_fn=None, lr=1., stop_grid_update_step=50, batch=-1,
                  small_mag_threshold=1e-16, small_reg_factor=1., metrics=None, sglr_avoid=False, save_fig=False, in_vars=None, out_vars=None, beta=3, save_fig_freq=1, img_folder='./video', device='cpu'):

        ''' Train with early stopping.

        Args:
        -----
        -- Changed --
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], 
                dataset['val_input'], dataset['val_label'], 
                dataset['test_input'], dataset['test_label']
        -- My par-s --
            tol : float
                Delta of validation fit which doesn`t count as fitness improvement. (Tolerence of training).
            n_iter_no_change : int
                Number of iteration with no fit change to early stopping.
        -----
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            stop_grid_update_step : int
                no grid updates after this training step
            batch : int
                batch size, if -1 then full.
            small_mag_threshold : float
                threshold to determine large or small numbers (may want to apply larger penalty to smaller numbers)
            small_reg_factor : float
                penalty strength applied to small factors relative to large factos
            device : str
                device   
            save_fig_freq : int
                save figure every (save_fig_freq) step

        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['val_loss'], 1D array of validation losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
        '''


        # Early stopping stuff preparation
        no_fit_change_steps = 0
        best_val_rmse = np.inf
        # Remembering first model
        best_model_dict = deepcopy(self.state_dict())

        def reg(acts_scale):

            def nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)


            reg_ = 0.
            for i in range(len(acts_scale)):
                vec = acts_scale[i].reshape(-1, )

                p = vec / torch.sum(vec)
                l1 = torch.sum(nonlinear(vec))
                entropy = - torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # both l1 and entropy

            # regularize coefficient to encourage spline to be zero
            for i in range(len(self.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(self.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(self.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        pbar = tqdm(range(steps), desc='description', ncols=130)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        results = {}
        results['train_loss'] = []
        results['val_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_val = dataset['val_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_val = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(device))
            if sglr_avoid == True:
                id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
            else:
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
            reg_ = reg(self.acts_scale)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

        # Main training loop
        for _ in pbar:

            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            val_id= np.random.choice(dataset['val_input'].shape[0], batch_size_val, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid:
                self.update_grid_from_samples(dataset['train_input'][train_id].to(device))

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(device))
                if sglr_avoid == True:
                    id_ = torch.where(torch.isnan(torch.sum(pred, dim=1)) == False)[0]
                    train_loss = loss_fn(pred[id_], dataset['train_label'][train_id][id_].to(device))
                else:
                    train_loss = loss_fn(pred, dataset['train_label'][train_id].to(device))
                reg_ = reg(self.acts_scale)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_loss = loss_fn_eval(self.forward(dataset['val_input'][val_id].to(device)), dataset['val_label'][val_id].to(device))
            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(device)), dataset['test_label'][test_id].to(device))

            # Early stopping processing stuff
            val_rmse = torch.sqrt(val_loss).cpu().detach().numpy()
            if (val_rmse > best_val_rmse - tol):
                no_fit_change_steps += 1
            else:
                no_fit_change_steps = 0

            if val_rmse < best_val_rmse:
                # Remembering best_val_fit and best_model
                best_val_rmse = val_rmse
                best_model_dict = deepcopy(self.state_dict())


            if _ % log == 0:
                pbar.set_description("trn_ls: %.2e | vl_ls: %.2e | e_stop: %d/%d | tst_ls: %.2e | reg: %.2e " % (
                                                        torch.sqrt(train_loss).cpu().detach().numpy(), 
                                                        val_rmse, 
                                                        no_fit_change_steps,
                                                        n_iter_no_change,
                                                        torch.sqrt(test_loss).cpu().detach().numpy(), 
                                                        reg_.cpu().detach().numpy() ))

            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['val_loss'].append(val_rmse)
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
            results['reg'].append(reg_.cpu().detach().numpy())

            if save_fig and _ % save_fig_freq == 0:
                self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                plt.close()

            # Checking early stopping criteria
            if no_fit_change_steps==n_iter_no_change:
                print(f'Early stopping criteria raised')
                break
        
        # Load best model
        self.load_state_dict(best_model_dict)
        self(dataset['train_input'])
        val_loss = loss_fn_eval(self.forward(dataset['val_input'][val_id].to(device)), dataset['val_label'][val_id].to(device))
        val_rmse = torch.sqrt(val_loss).cpu().detach().numpy()

        return results
    
    '''
    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        x_train_1, x_val, y_train_1, y_val = train_test_split(X, y, 
                                                    test_size=self.validation_fraction)
        
        val_dataset = {'train_input': torch.tensor(x_train_1, dtype=torch.float),
                   'train_label': torch.tensor(y_train_1, dtype=torch.float),
                   'val_input': torch.tensor(x_val, dtype=torch.float),
                   'val_label': torch.tensor(y_val, dtype=torch.float),
                   'test_input': torch.tensor(x_val, dtype=torch.float),
                   'test_label': torch.tensor(y_val, dtype=torch.float)}
    
        self.train_es(val_dataset, **self.train_kwargs)
        
        return self
    
    
    def predict(self, X):
        """Predict using the KAN_es model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        x = torch.tensor(X, dtype=torch.float)
        pred = self.forward(x).detach().numpy()
        return pred
    
    
    def get_params(self, deep=False):
        
        Get parameters
        
        return self.parameters()
    '''
    
    
def cross_val_KAN_es(X, y, 
                     d_kan_params, d_train_params,
                     validation_train_ration = 0.22,
                     cv = 5, l_func_metrics = [root_mean_squared_error, mean_absolute_error, r2_score]):
    '''
    Provides cross-validation to a KAN_es model.
    
    Args:
    -----
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        d_kan_params : dic
            dictionary for KAN_es initialisation
        d_train_params : dic
            dictionary for KAN_es training
        validation_train_ration : float            
            Validation / (Train+Validation)
        cv : int
            number of folds, seperating training and validation datasets
        l_func_metrics : List[callable]
            list of metrics for evaluating over test data.
            
    Returns:
        l_metrics : np.array
            2D matrix of evaluated metrics with shape [len(l_func_metrics), cv]
    '''
    
    kf = KFold(n_splits = cv)

    m_metrics = []
    

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        x_train = X[train_index]
        x_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        
        x_train_1, x_val, y_train_1, y_val = train_test_split(x_train, y_train, 
                                                    test_size=validation_train_ration,
                                                    random_state=i)
        
        kf_dataset = {'train_input': torch.tensor(x_train_1, dtype=torch.float),
                      'train_label': torch.tensor(y_train_1, dtype=torch.float),
                      'val_input': torch.tensor(x_val, dtype=torch.float),
                      'val_label': torch.tensor(y_val, dtype=torch.float),
                      'test_input': torch.tensor(x_test, dtype=torch.float),
                      'test_label': torch.tensor(y_test, dtype=torch.float)}

        model = KAN_es(**d_kan_params)
        print(f'kfold: {i}')
        results = model.train_es(kf_dataset, **d_train_params)

        test_pred = model.forward(kf_dataset['test_input']).detach().numpy() #.to(device)

        new_metrics = [metric(test_pred, kf_dataset['test_label']) for metric in l_func_metrics]
        m_metrics.append(new_metrics)
        
    model.plot()
    l_metrics = np.array(m_metrics).T


    return l_metrics


def kfold_experiment_es(l_X, l_y, l_d_kan_params, l_d_train_params, l_func_metrics=[root_mean_squared_error, mean_absolute_error, r2_score], cv = 5):
    '''
    Provides cross-validations to a KAN_es model to a list of model`s parameteres.
    
    Args:
    -----
        l_X : list of ndarrays or sparse matrix of shape (n_samples, n_features)
            The input data for each experiment.
        l_y : list of ndarrays of shape (n_samples,) or (n_samples, n_outputs)
            The target values for each experiment (class labels in classification, real numbers in
            regression).
        l_d_kan_params : List[dic]
            list of dictionaries for KAN_es initialisation
        l_d_train_params : List[dic]
            list of dictionaries for KAN_es training
        cv : int
            number of folds, seperating training and validation datasets
        l_func_metrics : List[callable]
            list of metrics for evaluating over test data.
            
    Returns:
        m_mean_metrics : np.array
            2D matrixes of evaluated metrics` means and std with shape [len(l_func_metrics), len(l_d_kan_params)].
    '''
    l_mean_metrics, l_std_metrics = [], []

    for X, y, d_kan_params, d_train_params in zip(l_X, l_y, l_d_kan_params, l_d_train_params):
        m_metrics = cross_val_KAN_es(X, y, d_kan_params, d_train_params, cv=cv)

        l_mean_metrics.append(m_metrics.mean(axis=1))
        l_std_metrics.append(m_metrics.std(axis=1))

    m_mean_metrics = np.array(l_mean_metrics).T
    m_std_metrics = np.array(l_std_metrics).T
    
    return m_mean_metrics, m_std_metrics


class KAN_es_cv(KAN_es):

    def set_train_kwargs(self, train_kwargs):
        """ Saves dictionary of KAN`s training parameteres for using in fit() method.
        """
        self.train_kwargs = train_kwargs
        
    def fit(self, X, y):
        """Fit the model to data matrix X and target(s) y.

        Parameters
        ----------
        X : ndarray or sparse matrix of shape (n_samples, n_features)
            The input data.

        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns a trained MLP model.
        """
        x_train_1, x_val, y_train_1, y_val = train_test_split(X, y, 
                                                    test_size=self.validation_fraction)
        
        val_dataset = {'train_input': torch.tensor(x_train_1, dtype=torch.float),
                   'train_label': torch.tensor(y_train_1, dtype=torch.float),
                   'val_input': torch.tensor(x_val, dtype=torch.float),
                   'val_label': torch.tensor(y_val, dtype=torch.float),
                   'test_input': torch.tensor(x_val, dtype=torch.float),
                   'test_label': torch.tensor(y_val, dtype=torch.float)}
    
        self.train_es(val_dataset, **self.train_kwargs)
        
        return self
    
    
    def predict(self, X):
        """Predict using the KAN_es model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        x = torch.tensor(X, dtype=torch.float)
        pred = self.forward(x).detach().numpy()
        return pred
    
    def get_params(self, deep=False):
        '''
        Get parameters
        '''
        return self.parameters()
