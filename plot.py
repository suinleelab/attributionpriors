import numpy as np
import matplotlib.pyplot as plt
import shap

colors = shap.plots.colors

def normalize(im, target_min=0.0, target_max=1.0):
    max_vals = np.amax(im, axis=(0,1,2), keepdims=True)
    min_vals = np.amin(im, axis=(0,1,2), keepdims=True)
    zero_one = (im - min_vals) / (max_vals - min_vals)
    return (zero_one * (target_max - target_min)) + target_min

def image_plot(shap_values, x, labels=None, show=True, width=20, aspect=0.2, hspace=0.2, take_abs=False, as_alpha=False, reverse=False, max_percentile=99.9, labelpad=None, class_names=None, bar=True, orig_image_overlay=True, orig_image_left=True):
    """ Plots SHAP values for image inputs.
    """

    multi_output = True
    if type(shap_values) != list:
        multi_output = False
        shap_values = [shap_values]

    # make sure labels
    if labels is not None:
        assert labels.shape[0] == shap_values[0].shape[0], "Labels must have same row count as shap_values arrays!"
        if multi_output:
            assert labels.shape[1] == len(shap_values), "Labels must have a column for each output in shap_values!"
        else:
            assert len(labels.shape) == 1, "Labels must be a vector for single output shap_values."

    label_kwargs = {} if labelpad is None else {'pad': labelpad}

    # plot our explanations
    fig_size = np.array([3 * (len(shap_values) + 1), 2.5 * (x.shape[0] + 1)])
    if fig_size[0] > width:
        fig_size *= width / fig_size[0]
    
    num_cols = len(shap_values)
    if orig_image_left:
        num_cols = num_cols + 1
    fig, axes = plt.subplots(nrows=x.shape[0], ncols=num_cols, figsize=fig_size)
    if len(axes.shape) == 1:
        axes = axes.reshape(1,axes.size)
    for row in range(x.shape[0]):
        x_curr = x[row].copy()

        # make sure
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 1:
            x_curr = x_curr.reshape(x_curr.shape[:2])
        if x_curr.max() > 1:
            x_curr /= 255.

        # get a grayscale version of the image
        if len(x_curr.shape) == 3 and x_curr.shape[2] == 3:
            x_curr_gray = (0.2989 * x_curr[:,:,0] + 0.5870 * x_curr[:,:,1] + 0.1140 * x_curr[:,:,2]) # rgb to gray
        else:
            x_curr_gray = x_curr

        if orig_image_left:
            axes[row,0].imshow(x_curr, cmap=plt.get_cmap('gray'))
            axes[row,0].axis('off')
        if len(shap_values[0][row].shape) == 2:
            abs_vals = np.stack([np.abs(shap_values[i]) for i in range(len(shap_values))], 0).flatten()
        else:
            abs_vals = np.stack([np.abs(shap_values[i].sum(-1)) for i in range(len(shap_values))], 0).flatten()
        max_val = np.nanpercentile(abs_vals, max_percentile)
        for i in range(len(shap_values)):
            if orig_image_left:
                index = i + 1
            else:
                index = i
            if labels is not None:
                axes[row,index].set_title(labels[row,i], **label_kwargs)
            sv = shap_values[i][row] if len(shap_values[i][row].shape) == 2 else shap_values[i][row].sum(-1)
            if take_abs:
                sv = np.abs(sv)
                min_val = np.min(sv)
                max_val = np.nanpercentile(sv, max_percentile)
                if reverse:
                    cmap='gray_r'
                else:
                    cmap='gray'
                im = axes[row,index].imshow(sv, cmap=cmap, vmin=min_val, vmax=max_val)
            elif as_alpha:
                alpha_map = np.abs(sv)
                alpha_map = np.clip(alpha_map, a_min=np.min(alpha_map), a_max=np.nanpercentile(alpha_map, max_percentile))
                alpha_map = np.expand_dims(alpha_map, axis=2)
                alpha_map = normalize(alpha_map)
                
                x_curr = normalize(x_curr)
                x_curr_alpha = np.concatenate([x_curr, alpha_map], axis=2)
                axes[row,index].imshow(x_curr_alpha, extent=(-1, sv.shape[0], sv.shape[1], -1))
            else:
                if orig_image_overlay:
                    axes[row,index].imshow(x_curr_gray, cmap=plt.get_cmap('gray'), alpha=0.15, extent=(-1, sv.shape[0], sv.shape[1], -1))
                im = axes[row,index].imshow(sv, cmap=colors.red_transparent_blue, vmin=-max_val, vmax=max_val)
            axes[row,index].axis('off')
    if hspace == 'auto':
        fig.tight_layout()
    else:
        fig.subplots_adjust(hspace=hspace)
    if not take_abs and not as_alpha and bar:
        cb = fig.colorbar(im, ax=np.ravel(axes).tolist(), label="Feature Importance", orientation="horizontal", aspect=fig_size[0]/aspect)
        cb.outline.set_visible(False)
    if show:
        plt.show()