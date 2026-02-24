import numpy as np
import os
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl

def save_sec_interest(result_xarr, experim_fold, 
                     figsize=(12,8)):
    '''save sections of interest named by TPU geophysicist
        1. Z-срез 2508 
        2. Z-срез 2568
        3. Z-срез 2608 XLine 2693 (оползень)
        4.Z-срез -2498, Inline 5784 (каналы)
        5. Z-срез -2480, Inline 5694 (каналы)'''
    
    pics_fold = os.path.join(experim_fold, 'cross_secs')
    # print(pics_fold)
    os.makedirs(pics_fold, exist_ok=True)

    def save_pic_only_twt(value):
        twt = value['twt']
        title = value['title']
        fig, ax = plt.subplots(1, figsize=figsize)
        im1 = ax.scatter(result_xarr.sel({'twt': twt}, method='nearest').cdp_x.data,
                result_xarr.sel({'twt': twt}, method='nearest').cdp_y.data,
                c=result_xarr.sel({'twt': twt}, method='nearest').data, 
                s=mpl.rcParams['lines.markersize']/4,
                rasterized=True)
        cbar = fig.colorbar(im1)
        ax.set_title(f'Twt: {twt}, {title}')
        # plt.show()
        fig.savefig(os.path.join(pics_fold, f'twt_{twt}_{title}.pdf'))
        # plt.show()
        plt.close()

    def save_pics_vert_hor(value):
        twt = value['twt']
        title = value['title']
        vert_sec_name = 'iline' if 'iline' in value.keys() else 'xline'
        vert_sec_val = value[vert_sec_name]
        another_vert_sec = 'cdp_x' if vert_sec_name == 'iline' else 'cdp_y'

        print(f'Vert_sec_name: {vert_sec_name}, vert_sec_val: {vert_sec_val} \n')
        print(f'another_vert_sec: {another_vert_sec}')

        fig, ax = plt.subplots(1, 2, figsize=(16,9))

        im_hor = ax[0].scatter(result_xarr.sel({'twt': twt}, 
                                               method='nearest').cdp_x.data,
                result_xarr.sel({'twt': twt}, 
                                method='nearest').cdp_y.data,
                c=result_xarr.sel({'twt': twt}, 
                                  method='nearest').data, 
                s=mpl.rcParams['lines.markersize']/4,
                rasterized=True)
        cbar = fig.colorbar(im_hor)
        

        ax[0].set_title(f'Twt: {twt}, {title}')

        ver_sec = result_xarr.sel({vert_sec_name: 
                                   vert_sec_val}, 
                                    method='nearest')

        im_vert = ax[1].imshow(
                                ver_sec.data.values.T,
                # s=mpl.rcParams['lines.markersize']/4,
                rasterized=True)

        cbar = fig.colorbar(im_vert)
        ax[1].set_title(f'{vert_sec_name}: {vert_sec_val}, {title}')
        fig.savefig(os.path.join(pics_fold, f'twt_{twt}_{vert_sec_name}_{vert_sec_val}_{title}.pdf'))
        # plt.show()
        plt.close()
        

    needed_secs = {1: {'twt': 2498, 'iline': 5784,
                       'title': 'Soil slip'},
                   
                   2: {'twt': 2568, 'title': 'Fans and channels'},
                   
                   3: {'twt': 2608, 'xline': 2693,
                       'title': 'Soil slip'},
                   
                   4: {'twt': 2508, 'title': 'Fans and channels'},
                   
                   5: {'twt': 2480, 'iline': 5694,
                       'title': 'Channels (not certainly)'}
                       }

    for val in needed_secs.values():
        print(f'start work with val: {val}')
        if ('iline' not in val.keys()) and ('xline' not in val.keys()):
            save_pic_only_twt(val)
        else:
            save_pics_vert_hor(val)