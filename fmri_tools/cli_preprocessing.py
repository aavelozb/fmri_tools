import argparse
from .filters import PreprocessVol

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess fMRI volumetric data (motion correction, masking, smoothing, filtering, and z-scoring)"
    )
    parser.add_argument('--func', type=str, required=True, help="Path to functional Nifti file")
    parser.add_argument('--mask', type=str, required=False, default=None, help="Path to brain mask Nifti file")
    parser.add_argument('--output-dir', type=str, default="results", help="Output directory")
    parser.add_argument('--TR0', type=float, required=True, help="Repetition time of original scan (seconds)")
    parser.add_argument('--TR', type=float, default=0.5, help="Target repetition time (seconds)")
    parser.add_argument('--n-first-scans-to-eliminate', type=int, default=0, help="Number of initial scans to discard")
    parser.add_argument('--no-parallel', action='store_true', help="Disable parallel processing")
    parser.add_argument('--n-jobs', type=int, default=-1, help="Number of parallel jobs (-1 uses all cores)")
    parser.add_argument('--no-verbose', action='store_true', help="Suppress progress information")

    args = parser.parse_args()

    preprocessor = PreprocessVol(
        TR0=args.TR0,
        fin_func=args.func,
        fin_mask=args.mask,
        output_dir=args.output_dir,
        TR=args.TR,
        n_first_scans_to_eliminate=args.n_first_scans_to_eliminate,
        verbose=not args.no_verbose,
        parallel=not args.no_parallel,
        n_jobs=args.n_jobs
    )
    preprocessor.fit()

if __name__ == "__main__":
    main()
