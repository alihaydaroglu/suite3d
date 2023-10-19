## 0.3  - 2023-10-19

### Added
- changelog
- benchmarking interface to test new updates and compare to previous outputs

## Changed
- output formats for cpu registration (job.register()) not match GPU registration
- Fusing integrated into CPU registration (similar to GPU reg)
- Split demo notebooks (w/ and w/out SVD denoising)

## Fixed
- CPU registration crashes 
- Correct padding during registration