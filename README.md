# msi_zarr_analysis

Analysis tools for Mass Spectrometry Images encoded as OME Zarr (see pims-plugin-format-msi)

## Note on slim datasets

- performances seem to be comparable. However, the automatic registration does not work well on these new images
  - how can they be improved ? the images aren't exactly the same...
  - the difference **possibly** comes from reduced noise which would alter the coloration...
  - is there a way to obtain the resolution of both images ?
    - **YES**: the resolution is given for all images (is it right tho ?)
  - can it be improved by using multiple lipids ? averaging over them should improve the stability of the scale

## notes binning algorithm

when there are few bins, the old implementation with only python is better
-> this is mostly marginal, and was probably initially observed due to random noise
-> the time when there are few bins don't matter much, so the new implementation can replace the old one !

### python mode, 512 bins

res_no.mean() = 8.827 res_no.std() = 0.043
res_.mean() = 4.472 res_.std() = 0.047

### python mode, 32 bins

res_no.mean() = 5.038 res_no.std() = 0.025
res_.mean() = 0.311 res_.std() = 0.001

### numba, 512 bins

-> repeat = 5
res_no.mean() = 0.271 res_no.std() = 0.350
res_.mean() = 0.353 res_.std() = 0.234
-> repeat = 25
res_no.mean() = 0.132 res_no.std() = 0.169
res_.mean() = 0.258 res_.std() = 0.109

### numba, 32 bins

-> repeat = 5
res_no.mean() = 0.215 res_no.std() = 0.350
res_.mean() = 0.151 res_.std() = 0.224
-> repeat = 25
res_no.mean() = 0.077 res_no.std() = 0.170
res_.mean() = 0.061 res_.std() = 0.110
