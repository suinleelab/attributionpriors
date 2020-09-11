#!/usr/bin/env python
import functools
import operator
import torch
from torch.autograd import grad
from torch.utils.data import DataLoader

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
    def __init__(self, background_dataset, batch_size, random_alpha=True,k=1):
        self.random_alpha = random_alpha
        self.k = k
        self.batch_size = batch_size
        self.ref_sampler = DataLoader(
                dataset=background_dataset, 
                batch_size=batch_size*k, 
                shuffle=True, 
                drop_last=True)
        return
    
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
            t_tensor = torch.FloatTensor(batch_size, k_).uniform_(0,1)

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

        grad_tensor = torch.zeros(samples_input.shape).float()

        
        for i in range(self.k):
            particular_slice = samples_input[:,i]
            batch_output = model(particular_slice)
            # should check that users pass in sparse labels
            # Only look at the user-specified label
            if batch_output.size(1) > 1:
                sample_indices = torch.arange(0,batch_output.size(0))
                indices_tensor = torch.cat([
                        sample_indices.unsqueeze(1), 
                        sparse_labels.unsqueeze(1)], dim=1)
                batch_output = gather_nd(batch_output, indices_tensor)

            model_grads = grad(
                    outputs=batch_output,
                    inputs=particular_slice,
                    grad_outputs=torch.ones_like(batch_output),
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
        reference_tensor = next(iter(self.ref_sampler))[0].float()
        shape = reference_tensor.shape
        reference_tensor = reference_tensor.view(
                self.batch_size, 
                self.k, 
                *(shape[1:]))
        samples_input = self._get_samples_input(input_tensor, reference_tensor)
        samples_delta = self._get_samples_delta(input_tensor, reference_tensor)
        grad_tensor = self._get_grads(samples_input, model, sparse_labels)
        mult_grads = samples_delta * grad_tensor
        expected_grads = mult_grads.mean(1)
        return expected_grads
