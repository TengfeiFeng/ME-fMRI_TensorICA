# ME-fMRI_TensorICA
Characterizing the distribution of neural and non-neural components in multi-echo EPI data across echo times based on Tensor-ICA
Here are the shared codes about how we decomposed ME-fMRI data using TensorICA. 

Then the codes in the '.py' were used to decompose the data based on the TensorICA implemented in MELODIC in the FSL toolbox. 
Here are examples of the three domains' distributions from one subject. And they were divided into three groups based on the distribution across TEs.
<img width="448" alt="image" src="https://github.com/TengfeiFeng/ME-fMRI_TensorICA/assets/48821629/ea51ca8f-a127-45b9-9e7e-f19e96e6c639">


Before decomposition,
we realigned the echoes' images separately using the motion parameters estimated from the first echo (13ms).
We smoothed the data with a 4mm FWHM kernel.
