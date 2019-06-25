import tensorflow as tf
import numpy as np

class AttributionPriorExplainer(object):
    def __init__(self, random_alpha=True):
        '''
        Returns an object of type AttributionPriorExplainer.
        Args:
            random_alpha: Whether or not the interpolation constant
                          should be drawn uniformly at random from U[0, 1], or
                          evenly spaced between [0, 1] with k points, where
                          k is the number of background reference used during training.
        Returns:
            The newly created object.'''
        self.random_alpha = random_alpha
        return
    
    def _permuted_references(self, input_tensor, k, shuffle=False):
        '''
        Given a tensor of shape [None, ...] where ... indicates input dimensions,
        returns a random shuffling of that input.
        Args:
            input_tensor: An input tensor of shape [None, ...], where ...
                          indicates the input dimensions.
            return_func: Whether or not to return a function as opposed to a tensor.
            shuffle: If true, shuffles randomly along the batch axis. If false, 
                     rolls deterministically along the batch axis.
        Returns:
            A tensor of shape [None, k, ...]. A background reference operation.
        '''
        if shuffle:
            shuffle_list = [tf.random.shuffle(input_tensor)] * k
        else:
            shuffle_list = [tf.manip.roll(input_tensor, shift=i, axis=0) for i in range(1, k+1)]
        return tf.stack(shuffle_list, axis=1, name='background_ref_op')
    
    def _grad_across_multi_output(self, output_tensor, input_tensor, sparse_labels_op=None):
        '''
        Calculates the gradients for each output with respect to each input.
        Args:
            input_tensor: An input tensor of shape [None, ...] where ...
                          indicates the input dimensions. This function will throw an
                          error if input_tensor is of type list.
            output_tensor: A tensor indicating the output of the model. Should be shaped as
                           [None, num_classes] where None indicates the batch dimension
                           and num_classes is the number of output classes of the model.
        Returns:
            A tensor of shape [None, num_classes, ...], where the ... indicates the input dimensions.
        '''
        if sparse_labels_op is not None:
            sample_indices = tf.range(tf.shape(output_tensor)[0])
            
            indices_tensor = tf.stack([sample_indices, tf.cast(sparse_labels_op, tf.int32)], axis=1)
                
            class_selected_output_tensor = tf.gather_nd(output_tensor, indices_tensor)
            
            class_selected_output_tensor.set_shape([None])
            grad_tensor = tf.gradients(class_selected_output_tensor, input_tensor)[0]
            return grad_tensor
        else:
            output_class_tensors = tf.unstack(output_tensor, axis=1)
            grad_tensors = []
            for output_class_tensor in output_class_tensors:
                grad_tensor = tf.gradients(output_class_tensor, input_tensor)[0]
                grad_tensors.append(grad_tensor)
            multi_grad_tensor = tf.stack(grad_tensors, axis=1)
            return multi_grad_tensor    
    
    def input_to_samples_delta(self, batch_input_op, background_ref_op='roll', k=1):
        '''
        Creates graph operations to switch between normal input and 
        input-reference interpolations. 

        Args:
            batch_input_op: Tensor of shape (None, ...), where ... indicates
                            the input dimensions. 
            background_ref_op: A tensor of shape (None, k, ...) where ... indicates 
                               the input dimensions, and k represents the number of
                               background reference samples to draw per input in the batch,
                               or a function that returns such a tensor when called. Alternatively,
                               if set to the string 'shuffle' or 'roll', 
                               uses a shuffled/rolled version of the batch input as a background reference.
            k: An integer specifying the number of background references if background_ref_op is 'roll' or
               'shuffle'. Ignored if background_ref_op is a tensor.
        Returns: 
            cond_input_op: A tensor of shape (None, ...). Use this tensor to build your model.
            train_eg: A placeholder with default value False. Set to true to switch from normal inputs
                      to interpolated reference inputs.
        '''
        input_dims = batch_input_op.get_shape().as_list()[1:]
        num_input_dims = len(input_dims)
            
        def samples_input_fn():
            if tf.contrib.framework.is_tensor(background_ref_op):
                self.background_ref_op = background_ref_op
                if k is not None:
                    print("Warning: value `{}` of parameter k will be ignored because background_ref_op was an input tensor".format(k))
            elif callable(background_ref_op):
                self.background_ref_op = background_ref_op()
                if k is not None:
                    print("Warning: value `{}` of parameter k will be ignored because background_ref_op was callable".format(k))
            elif isinstance(background_ref_op, str):
                if background_ref_op == 'shuffle':
                    self.background_ref_op = self._permuted_references(batch_input_op, k, shuffle=True)
                elif background_ref_op == 'roll':
                    self.background_ref_op = self._permuted_references(batch_input_op, k, shuffle=False)
                else:
                    raise ValueError('Unrecognized string value `{}` for parameter `background_ref_op` (must be one of `shuffle`, `roll`)'.format(background_ref_op))
            else:
                raise ValueError('Unrecognized value `{}` for parameter `background_ref_op`'.format(background_ref_op))

            batch_size = tf.shape(self.background_ref_op)[0]
            k_ = self.background_ref_op.shape[1]

            #Grab a [batch_size, k]-sized interpolation sample
            #Note that evaluating t_tensor will evaluate background_ref_op implicitly for shape information
            if self.random_alpha:
                t_tensor = tf.random_uniform(shape=[batch_size, k_], name='t_tensor')
                t_tensor = tf.cast(t_tensor, dtype=batch_input_op.dtype)
                t_tensor.set_shape([None, k_])
            else:
                t_tensor = tf.linspace(start=0.0, stop=1.0, num=k, name='linspace_t')
                t_tensor = tf.expand_dims(t_tensor, axis=0)
                t_tensor = tf.tile(t_tensor, [batch_size, 1], name='t_tensor')
                t_tensor = tf.cast(t_tensor, dtype=batch_input_op.dtype)
                t_tensor.set_shape([None, k_])
                
            interp_coef = tf.reshape(t_tensor, [batch_size, k_] + [1] * num_input_dims, name='interp_coef')

            #Evaluate the end points
            end_point_ref = tf.multiply(1.0 - interp_coef, self.background_ref_op, name='end_point_ref')

            input_expand_mult = tf.expand_dims(batch_input_op, axis=1)
            end_point_input = tf.multiply(interp_coef, input_expand_mult, name='end_point_input')

            #Add the end points together, because, you know, interpolation
            samples_input = tf.add(end_point_input, end_point_ref, name='samples_input')
            return samples_input

        #Define operations to switch between interpolation and normal input
        train_eg = tf.placeholder_with_default(False, shape=(), name='train_eg')
        cond_input_op = tf.cond(train_eg, samples_input_fn, lambda: batch_input_op)
        cond_input_op = tf.reshape(cond_input_op, shape=[-1] + input_dims, name='cond_input_op')
        
        self.samples_delta = tf.subtract(tf.expand_dims(batch_input_op, axis=1), 
                               self.background_ref_op, name='samples_delta')

        return cond_input_op, train_eg

    def shap_value_op(self, output_op, cond_input_op, sparse_labels_op=None):
        '''
        Creates a tensor operation to calculate expected gradients with respect to the provided input operation.
        This will throw an error if you haven't first called input_to_samples_delta.
        
        Args:
            output_op: The output layer of your model, or the layer to take the expected gradients with respect to.
            cond_input_op: An operation that returns interpolations between inputs and background reference samples.
                           The output from input_to_samples_delta.
            sparse_labels_op: For multi-class problems, the true labels of your input data. Assumes that there are a discrete
                              number of classes from 0 to num_classes and this tensor provides the integer label (NOT ONE HOT ENCODED)
                              for each batch of input. This tensor is used to index into the gradient operation such that the
                              explanations returned for each sample are of its true class. You can also manipulate 
                              this operation to get explanations for a specific class, or set it to 
                              None to get an operation that returns explanations for multiple classes.
        Returns:
            expected_grads: A tensor of shape (None, ...), the same shape as an input batch. The expected gradients with 
                            respect to the input batch.
        '''
        multi_output = True
        if len(output_op.shape) == 1:
            multi_output = False
            
        refs_per_input = self.samples_delta.get_shape().as_list()[1]
    
        if multi_output:
            if sparse_labels_op is None:
                print('You have requested multi-class values, but have not provided a labels tensor. This may be memory intensive...')
                
                #Of shape (None, num_classes, ...)
                gradient_tensor = self._grad_across_multi_output(output_op, cond_input_op)
                gradient_tensor = tf.reshape(gradient_tensor, shape=[-1, refs_per_input, output_op.shape[-1]] + cond_input_op.get_shape().as_list()[1:], name='gradient_tensor')
                mult_grads = tf.expand_dims(self.samples_delta, axis=2)
                mult_grads = tf.multiply(mult_grads, gradient_tensor, name='mult_grads')
                expected_grads = tf.reduce_mean(mult_grads, axis=1)
                return expected_grads
            
            with tf.device('/cpu:0'):
                sparse_labels_op = tf.tile(tf.expand_dims(sparse_labels_op, axis=1), (1, refs_per_input))
                sparse_labels_op = tf.reshape(sparse_labels_op, (tf.shape(cond_input_op)[0], ), name='sparse_labels_op')

            #Gradients will be of shape (None, ...)
            gradient_tensor =  self._grad_across_multi_output(output_op, cond_input_op, sparse_labels_op)
        else:
            assert sparse_labels_op is None, 'You have passed in a sparse_labels_op, but your model is not multi-output'
            #Gradients will be of shape (None, ...) - same shape as input
            gradient_tensor = tf.gradients(output_op, cond_input_op)[0]
        
        #Reshape the gradient tensor into (None, k, ...) so that we can average across the k.
        #Note: it is important that the axis ordering is (None, k, ...). Keeping the k before
        #the feature dimensions allows the reshaping after applying the model output to behave correctly. 
        #This is because the right-most dimensions are squashed first when reshaping.
        gradient_tensor = tf.reshape(gradient_tensor, shape=[-1, refs_per_input] + gradient_tensor.get_shape().as_list()[1:], name='gradient_tensor') 

        #Multiply gradients and input-reference difference
        mult_grads = tf.multiply(self.samples_delta, gradient_tensor, name='mult_grads')
    
        #Average over k, the background references
        expected_grads = tf.reduce_mean(mult_grads, axis=1, name='expected_grads')
        return expected_grads