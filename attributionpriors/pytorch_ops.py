#!/usr/bin/env python
import functools
import operator
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

DEFAULT_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def gather_nd(params, indices):
    """
    Args:
        params: Tensor to index
        indices: k-dimension tensor of integers. 
    Returns:
        output: 1-dimensional tensor of elements of ``params``, where
            output[i] = params[i][indices[i]]

            params   indices   output

            1 2       1 1       4
            3 4       2 0 ----> 5
            5 6       0 0       1
    """
    max_value = functools.reduce(operator.mul, list(params.size())) - 1
    indices = indices.t().long()
    ndim = indices.size(0)
    idx = torch.zeros_like(indices[0]).long()
    m = 1

    for i in range(ndim)[::-1]:
        idx += indices[i]*m
        m *= params.size(i)

    idx[idx < 0] = 0
    idx[idx > max_value] = 0
    return torch.take(params, idx)
    

class AttributionPriorExplainer(object):
    def __init__(self, background_dataset, batch_size, random_alpha=True,k=1,scale_by_inputs=True):
        self.random_alpha = random_alpha
        self.k = k
        self.scale_by_inputs = scale_by_inputs
        self.batch_size = batch_size
        self.ref_set = background_dataset
        self.ref_sampler = DataLoader(
                dataset=background_dataset, 
                batch_size=batch_size*k, 
                shuffle=True, 
                drop_last=True)
        return

    def _get_ref_batch(self,k=None):
        return next(iter(self.ref_sampler))[0].float()
    
    def _get_samples_input(self, input_tensor, reference_tensor):
        '''
        calculate interpolation points
        Args:
            input_tensor: Tensor of shape (batch, ...), where ... indicates
                          the input dimensions. 
            reference_tensor: A tensor of shape (batch, k, ...) where ... 
                indicates dimensions, and k represents the number of background 
                reference samples to draw per input in the batch.
        Returns: 
            samples_input: A tensor of shape (batch, k, ...) with the 
                interpolated points between input and ref.
        '''
        input_dims = list(input_tensor.size())[1:]
        num_input_dims = len(input_dims)
            
        batch_size = reference_tensor.size()[0]
        k_ = reference_tensor.size()[1]

        # Grab a [batch_size, k]-sized interpolation sample
        if self.random_alpha:
            t_tensor = torch.FloatTensor(batch_size, k_).uniform_(0,1).to(DEFAULT_DEVICE)
        else:
            if k_==1:
                t_tensor = torch.cat([torch.Tensor([1.0]) for i in range(batch_size)]).to(DEFAULT_DEVICE)
            else:
                t_tensor = torch.cat([torch.linspace(0,1,k_) for i in range(batch_size)]).to(DEFAULT_DEVICE)

        shape = [batch_size, k_] + [1] * num_input_dims
        interp_coef = t_tensor.view(*shape)

        # Evaluate the end points
        end_point_ref = (1.0 - interp_coef) * reference_tensor

        input_expand_mult = input_tensor.unsqueeze(1)
        end_point_input = interp_coef * input_expand_mult
        
        # A fine Affine Combine
        samples_input = end_point_input + end_point_ref
        return samples_input
    
    def _get_samples_delta(self, input_tensor, reference_tensor):
        input_expand_mult = input_tensor.unsqueeze(1)
        sd = input_expand_mult - reference_tensor
        return sd
    
    def _get_grads(self, samples_input, model, sparse_labels=None):
        samples_input.requires_grad = True

        grad_tensor = torch.zeros(samples_input.shape).float().to(DEFAULT_DEVICE)

        
        for i in range(self.k):
            particular_slice = samples_input[:,i]
            batch_output = model(particular_slice)
            # should check that users pass in sparse labels
            # Only look at the user-specified label
            if batch_output.size(1) > 1:
                sample_indices = torch.arange(0,batch_output.size(0)).to(DEFAULT_DEVICE)
                indices_tensor = torch.cat([
                        sample_indices.unsqueeze(1), 
                        sparse_labels.unsqueeze(1)], dim=1)
                batch_output = gather_nd(batch_output, indices_tensor)

            model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output).to(DEFAULT_DEVICE),
                    create_graph=True)
            grad_tensor[:,i,:] = model_grads[0]
        return grad_tensor
           
    def shap_values(self, model, input_tensor, sparse_labels=None):
        """
        Calculate expected gradients approximation of Shapley values for the 
        sample ``input_tensor``.

        Args:
            model (torch.nn.Module): Pytorch neural network model for which the
                output should be explained.
            input_tensor (torch.Tensor): Pytorch tensor representing the input
                to be explained.
            sparse_labels (optional, default=None): 
        """
        reference_tensor = self._get_ref_batch()
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
                self.batch_size, 
                self.k, 
                *(shape[1:])).to(DEFAULT_DEVICE)
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)
        mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        expected_grads = mult_grads.mean(1)
        return expected_grads

class VariableBatchExplainer(AttributionPriorExplainer):
    """
    Subclasses AttributionPriorExplainer to avoid pre-specified batch size. Will adapt batch
    size based on shape of input tensor.
    """
    def __init__(self, background_dataset, random_alpha=True,scale_by_inputs=True):
        """
        Arguments:
        background_dataset: PyTorch dataset - may not work with iterable-type (vs map-type) datasets
        random_alpha: boolean - Whether references should be interpolated randomly (True, corresponds
            to Expected Gradients) or on a uniform grid (False - corresponds to Integrated Gradients)
        """
        self.random_alpha = random_alpha
        self.k = None
        self.scale_by_inputs=scale_by_inputs
        self.ref_set = background_dataset
        self.ref_sampler = DataLoader(
                dataset=background_dataset, 
                batch_size=1, 
                shuffle=True, 
                drop_last=True)
        self.refs_needed = -1
        return

    def _get_ref_batch(self,refs_needed=None):
        """
        Arguments:
        refs_needed: int - number of references to provide
        """
        if refs_needed!=self.refs_needed:
            self.ref_sampler = DataLoader(
                dataset=self.ref_set, 
                batch_size=refs_needed, 
                shuffle=True, 
                drop_last=True)
            self.refs_needed = refs_needed
        return next(iter(self.ref_sampler))[0].float()
           
    def shap_values(self, model, input_tensor, sparse_labels=None,k=1):
        """
        Arguments:
        base_model: PyTorch network
        input_tensor: PyTorch tensor to get attributions for, as in normal torch.nn.Module API
        sparse_labels:  np.array of sparse integer labels, i.e. 0-9 for MNIST. Used if you only
            want to explain the prediction for the true class per sample.
        k: int - Number of references to use default for explanations. As low as 1 for training.
            100-200 for reliable explanations. 
        """
        self.k = k
        n_input = input_tensor.shape[0]
        refs_needed = n_input*self.k
        # This is a reasonable check but prevents compatibility with non-Map datasets
        assert refs_needed<=len(self.ref_set), "Can't have more samples*references than there are reference points!"
        reference_tensor = self._get_ref_batch(refs_needed)
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
                n_input, 
                self.k,
                *(shape[1:])).to(DEFAULT_DEVICE)
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)
        mult_grads = samples_delta * grad_tensor if self.scale_by_inputs else grad_tensor
        expected_grads = mult_grads.mean(1)

        return expected_grads

class ExpectedGradientsModel(torch.nn.Module):
    """
    Wraps a PyTorch model (one that implements torch.nn.Module) so that model(x)
    produces SHAP values as well as predictions (controllable by 'shap_values'
    flag.
    """
    def __init__(self,base_model,refset,k=1,random_alpha=True,scale_by_inputs=True):
        """
        Arguments:
        base_model: PyTorch network that subclasses torch.nn.Module
        refset: PyTorch dataset - may not work with iterable-type (vs map-type) datasets
        k: int - Number of references to use by default for explanations. As low as 1 for training.
            100-200 for reliable explanations. 
        """
        super(ExpectedGradientsModel,self).__init__()
        self.k = k
        self.base = base_model
        self.refset = refset
        self.random_alpha = random_alpha
        self.exp = VariableBatchExplainer(self.refset,random_alpha=random_alpha,scale_by_inputs=scale_by_inputs)
    def forward(self,x,shap_values=False,sparse_labels=None,k=1):
        """
        Arguments:
        x: PyTorch tensor to predict with, as in normal torch.nn.Module API
        shap_values:     Binary flag -- whether to produce SHAP values
        sparse_labels:  np.array of sparse integer labels, i.e. 0-9 for MNIST. Used if you only
            want to explain the prediction for the true class per sample.
        k: int - Number of references to use default for explanations. As low as 1 for training.
            100-200 for reliable explanations. 
        """
        output = self.base(x)
        if not shap_values: return output
        else: shaps = self.exp.shap_values(self.base,x,sparse_labels=sparse_labels,k=k)
        return output, shaps