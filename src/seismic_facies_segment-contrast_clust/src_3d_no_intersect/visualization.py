import numpy as np
import os 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def save_pic_only_horizon(result_xarr, pic_config, figsize=(16,9)):
    
    def get_surf_data(surf, sect_name, sect_ind): 
        curve = surf[surf[sect_name]==sect_ind]
        return curve
    cube_xarray = result_xarr.copy()
    # pics_fold = os.path.join(experim_fold, 'cross_secs')
    # title = value['title']
    
    vert_sec_name = 'iline' if 'iline' in pic_config.keys() else 'xline'
    orthog_direct = 'iline' if vert_sec_name == 'xline' else 'xline'
    vert_sec_val = pic_config[vert_sec_name]
    # another_vert_sec = 'cdp_x' if vert_sec_name == 'iline' else 'cdp_y'
    fig, ax = plt.subplots(1, figsize=figsize)
    # plt.show()
    
    ver_sec = cube_xarray.sel({vert_sec_name: 
                                vert_sec_val}, 
                                method='nearest')
    xtick_labels = ver_sec['iline'].values
    ytick_labels = ver_sec.twt.values

    ver_sec = ver_sec.data.values.copy()
    ver_sec = ver_sec.astype(float)
    ver_sec[ver_sec == -1] = np.nan
    ver_sec = ver_sec.T

    unique_classes = np.unique(ver_sec[~np.isnan(ver_sec)])
    unique_classes = np.sort(unique_classes)  # Ensure consistent ordering
    idx_to_class = {idx: orig_val for idx, orig_val in enumerate(unique_classes)}
    n_classes = len(unique_classes)
    base_cmap = plt.get_cmap('tab10')
    
    if hasattr(base_cmap, 'colors') and len(base_cmap.colors) >= n_classes:
        # If colormap has discrete colors, use them directly
        colors = base_cmap.colors[:n_classes]
    else:
        # Sample colors from continuous colormap
        colors = [base_cmap(i / n_classes) for i in range(n_classes)]
    
    # Create discrete colormap
    discrete_cmap = ListedColormap(colors)

    im_vert = ax.pcolormesh(xtick_labels, ytick_labels,
                        ver_sec, cmap=discrete_cmap,
            # s=mpl.rcParams['lines.markersize']/4,
            rasterized=True)

    cbar = plt.colorbar(im_vert, ax=ax, extend='neither', shrink=0.5)
        
        # Set ticks at the mapped indices (0, 1, 2, ...)
    tick_positions = np.arange(n_classes)
    cbar.set_ticks(tick_positions)
    tick_labels = [f'{int(idx_to_class[i])}' for i in range(n_classes)]
        
    cbar.set_ticklabels(tick_labels)
    # ax.set_xticklabels(xtick_labels)
    # ax.set_yticklabels(ytick_labels)
    plt.gca().invert_yaxis()
    ax.set_aspect(1.5)
    ax.set_title(f'{vert_sec_name}: {vert_sec_val}', fontsize=15)
    
    if 'surfaces' in pic_config.keys():
        surf_dct = pic_config['surfaces']
        for surf_name, surf_df in surf_dct.items():
            curve = get_surf_data(surf_df, vert_sec_name, vert_sec_val)
            c = 'black' if 'reflect' in surf_name.lower() else 'red'
            if 'upper' in surf_name.lower():
                xmin, xmax = curve[orthog_direct].min(), curve[orthog_direct].max()
            ax.plot(curve[orthog_direct], curve['twt'], color=c, label=surf_name)
    ax.set_xlim(xmin - 20, xmax + 20)
    ax.legend()    

    ax.set_xlabel(orthog_direct, fontsize=12)
    ax.set_ylabel('twt', fontsize=12)

    # fig.savefig(os.path.join(pics_fold, f'{vert_sec_name}_{vert_sec_val}.pdf'))
    # plt.show()
    # plt.close()
    return fig, ax