import os
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nb
from pathlib import Path
from typing import Optional, Union
from rich import print

from scipy import stats
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from tqdm import tqdm

from nilearn.image import resample_to_img, mean_img, smooth_img, clean_img
from nilearn.masking import apply_mask, unmask
import warnings
warnings.filterwarnings("ignore")

def linear_interpolate_signals(Y0, timeline0, timeline, parallel=False, verbose=False, n_jobs=-1):
    def process_single_signal(y):
        return interp1d(timeline0, y, 
                        kind='linear', bounds_error=False, 
                        fill_value='extrapolate')(timeline)

    desc = 'Interpolation'
    if verbose:
        start_time = time.time()

    if Y0.ndim == 1:
        return process_single_signal(Y0)
    elif Y0.ndim == 2:
        N = Y0.shape[1]
        if parallel:
            Yinterp = Parallel(n_jobs=n_jobs)(
                delayed(process_single_signal)(y) for y in tqdm(Y0.T, total=N, desc=f'{desc} in parallel', disable=not verbose)
            )
        else:
            Yinterp = [process_single_signal(y) for y in tqdm(Y0.T, total=N, desc=desc, disable=not verbose)]

        if verbose:
            print(f'Elapsed time for interpolating {N} signals: {time.time() - start_time:.4f} seconds')

        return np.array(Yinterp).T
    else:
        raise ValueError('Input signals must be 1D or 2D numpy array')

def rest_IdealFilter(Y, TR, bands, parallel=False, verbose=False, n_jobs=-1):
    """
    Applies an ideal bandpass filter on fMRI data with optional parallel processing.
    """
    def process_single_signal(s):
        Ny = s.shape[0]
        fy = fft(np.concatenate((s, np.flipud(s))))
        f = fftfreq(fy.shape[0], TR)
        low, high = bands
        mask = (np.abs(f) >= low) & (np.abs(f) < high)
        fy[~mask] = 0
        y = np.real(ifft(fy))[:Ny]
        return y + np.mean(s)

    desc = 'Ideal Filtering'
    if verbose:
        start_time = time.time()

    if Y.ndim == 1:
        return process_single_signal(Y)
    elif Y.ndim == 2:
        N = Y.shape[1]
        if parallel:
            Y_filtered = Parallel(n_jobs=n_jobs)(
                delayed(process_single_signal)(s) for s in tqdm(Y.T, total=N, desc=f'{desc} in parallel', disable=not verbose)
            )
        else:
            Y_filtered = [process_single_signal(s) for s in tqdm(Y.T, total=N, desc=desc, disable=not verbose)]
        if verbose:
            print(f'Elapsed time for filtering {N} signals: {time.time() - start_time:.4f} seconds')
        return np.array(Y_filtered).T
    else:
        raise ValueError('Input signals must be 1D or 2D numpy array')

class PreprocessVol:
    """
    Preprocess volumetric fMRI data including motion correction, masking, smoothing,
    temporal filtering, and z-scoring. Processes can run in parallel with progress shown.
    """
    def __init__(
        self,
        TR0: Union[int, float],
        fin_func: str,
        fin_mask: Optional[str] = None,
        output_dir: str = 'results',
        TR: Union[int, float] = 0.5,
        n_first_scans_to_eliminate: int = 0,
        verbose: bool = True,
        parallel: bool = True,
        n_jobs: int = -1):

        self.fin_func = fin_func
        self.TR0 = TR0
        self.TR = TR
        self.fin_mask = fin_mask
        self.n_first_scans_to_eliminate = n_first_scans_to_eliminate
        self.output_dir = output_dir
        self.verbose = verbose
        self.parallel = parallel
        self.n_jobs = n_jobs

    def fit(self):
        if self.verbose:
            print(f'[bold green]Running preprocessing on {self.fin_func}')
            if self.fin_mask is not None:
                print(f'Mask input: {self.fin_mask}')

        ####### making output directory
        os.makedirs(self.output_dir, exist_ok=True)

        Y0_img = nb.load(self.fin_func)
        Y0_img_motion_corrected = clean_img(Y0_img, detrend=True, standardize=False, t_r=self.TR0)
        anat_img = mean_img(smooth_img(Y0_img_motion_corrected, 5))

        if self.fin_mask is not None:
            mask_img = nb.load(self.fin_mask, mmap=True)
            mask_img = nb.Nifti1Image((mask_img.get_fdata() > 0).astype(int),
                                       mask_img.affine, mask_img.header)
        else:
            mask_img = nb.Nifti1Image(np.ones(anat_img.shape[:3], dtype=int),
                                       anat_img.affine, anat_img.header)

        Y0_motion_corrected = apply_mask(Y0_img_motion_corrected, mask_img)
        Y0_firstn_eliminated = Y0_motion_corrected[self.n_first_scans_to_eliminate:, :]
        timeline0 = np.arange(Y0_firstn_eliminated.shape[0]) * self.TR0

        if self.verbose:
            print(f'Number of signals to be processed: {Y0_motion_corrected.shape[1]}')
            print(f'Original number of time points: {Y0_motion_corrected.shape[0]}')
            print(f'{self.n_first_scans_to_eliminate} first scans eliminated')
            print(f'New number of time points: {Y0_firstn_eliminated.shape[0]}')
            print(f'TR original {self.TR0} s - Time span: {timeline0[0]:.2f}-{timeline0[-1]:.2f}')

        Y0_img_smooth3D = smooth_img(unmask(Y0_firstn_eliminated, mask_img), fwhm=6)
        Y0_smooth3D = apply_mask(Y0_img_smooth3D, mask_img)

        # Interpolation to lower TR if required
        if self.TR < self.TR0:
            T = int((self.TR0 / self.TR) * (Y0_smooth3D.shape[0] - 1)) + 1
            timeline = np.arange(T) * self.TR
            Y = linear_interpolate_signals(
                Y0_smooth3D,
                timeline0,
                timeline,
                parallel=self.parallel,
                verbose=self.verbose,
                n_jobs=self.n_jobs
            )
            if self.verbose:
                print(f'TR high resolution {self.TR} s - Time span: {timeline[0]:.2f}-{timeline[-1]:.2f}')
        else:
            timeline = timeline0.copy()
            Y = Y0_smooth3D.copy()

        ################### Temporal filtering and z-score normalization
        Y_tfilter = rest_IdealFilter(
            Y,
            self.TR,
            bands=[0.01, 0.08],
            parallel=self.parallel,
            verbose=self.verbose,
            n_jobs=self.n_jobs
        )
        Y_img_smooth3D = smooth_img(unmask(Y_tfilter, mask_img), fwhm=6)
        Y_smooth3D = apply_mask(Y_img_smooth3D, mask_img)
        Y_normalized = stats.zscore(Y_smooth3D, axis=0, ddof=1)
        Y_img_normalized = unmask(Y_normalized, mask_img)

        basename = os.path.basename(self.fin_func)
        fout_func = os.path.join(self.output_dir, f'preproc_{basename}')

        if self.verbose:
            print('Temporal/spatial filtering and interpolation done.')
            print('[bold yellow]=======> saving preprocessed signals on:')
            print(fout_func)

        nb.save(Y_img_normalized, fout_func)







if __name__ == "__main__":

    from nilearn import datasets

    dataset = datasets.fetch_abide_pcp(SUB_ID = [51461], 
                                    derivatives = ['func_mask', 'func_mean', 'reho', 'func_preproc'],
                                    data_dir = os.path.join('.','testdata'))

    fin_func = os.path.join('testdata', 'ABIDE_pcp', 'cpac', 'nofilt_noglobal', 'Caltech_0051461_func_preproc.nii.gz')
    fin_mask = os.path.join('testdata', 'ABIDE_pcp', 'cpac', 'nofilt_noglobal', 'Caltech_0051461_func_mask.nii.gz')

    # preprocessor = PreprocessVol(fin_func=fin_func, fin_mask=fin_mask, TR0=2, n_first_scans_to_eliminate=5)
    # preprocessor.fit()
    
    
   
    # TR0 = 2
    # TR = 0.5
    # n_first_scans_to_eliminate = 5
    # Y0_img = nb.load(fin_func)

    # mask_img = nb.load(fin_mask, mmap=True)
    # mask_data = mask_img.get_fdata() > 0  # shape: x, y, z

    # Y0_img_motion_corrected = clean_img(Y0_img, detrend=True, standardize=False, t_r=TR0)
    # anat_img = mean_img(smooth_img(Y0_img_motion_corrected, 5))

    # coords = np.array(np.where(mask_data)).T  # N_vox, 3
    # n_voxels = coords.shape[0]
    # random_voxel_idx = np.random.randint(0, n_voxels)
    # x, y, z = coords[random_voxel_idx]


    # Y0_motion_corrected = apply_mask(Y0_img_motion_corrected, mask_img)
    # signal0 = Y0_img_motion_corrected.get_fdata()[x, y, z, n_first_scans_to_eliminate:]
    # Y0_firstn_eliminated = Y0_motion_corrected[n_first_scans_to_eliminate:, :]
    # timeline0 = np.arange(Y0_firstn_eliminated.shape[0]) * TR0

    

    # plt.figure(figsize=(12, 6))
    # plt.plot(timeline0, signal0, label='Original Signal (includes eliminated)')
    # # plt.axvspan(timeline_original[0],
    # #             timeline_original[len(timeline_original) - len(timeline_processed) - 1],
    # #             color='gray', alpha=0.3, label='Eliminated Samples')


    # Y0_img_smooth3D = smooth_img(unmask(Y0_firstn_eliminated, mask_img), fwhm=6)
    # signal0_smooth3D = Y0_img_smooth3D.get_fdata()[x, y, z, :] 
    # Y0_smooth3D = apply_mask(Y0_img_smooth3D, mask_img)

    # plt.plot(timeline0, signal0_smooth3D, label='Original Signal (includes eliminated)')

    # T = int((TR0 / TR) * (Y0_smooth3D.shape[0] - 1)) + 1
    # timeline = np.arange(T) * TR
    # Y = linear_interpolate_signals(
    #     Y0_smooth3D,
    #     timeline0,
    #     timeline,
    #     parallel=True,
    #     verbose=True,
    #     n_jobs=-1)




    # ################### Temporal filtering and z-score normalization
    # Y_tfilter = rest_IdealFilter(Y, TR, bands=[0.01, 0.08])
    # Y_img_smooth3D = smooth_img(unmask(Y_tfilter, mask_img), fwhm=6)
    # Y_smooth3D = apply_mask(Y_img_smooth3D, mask_img)
    # Y_normalized = stats.zscore(Y_smooth3D, axis=0, ddof=1)
    # Y_img_normalized = unmask(Y_normalized, mask_img)

    # signal1 = Y_img_smooth3D.get_fdata()[x, y, z, :] 
    # plt.plot(timeline, signal1, label='result')
    # plt.show()


    # plt.plot(timeline_processed, processed_signal, label='Processed Signal (after elimination)')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Signal Intensity')
    # plt.title(f'Sample Signal Comparison (Signal index {index})')
    # plt.legend()
    # plt.tight_layout()
    # save_path = os.path.join(output_dir, f'sample_signal_{index}.png')
    # plt.savefig(save_path)
    # plt.close()
    # print(f'Saved figure to {save_path}')
