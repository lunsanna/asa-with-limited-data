Band reject
Instead of
- sample the starting frequency $f_0$ uniformly in the mel scale 
- transforme $f_0$ to the linear scale 
- define frequency masking range as $[f_0, f_0 + mask_width)$ (mask_width in linear scale)
Do the following
- sample the start frequency $f_0$ uniformly in the mel scale 
- define the end frequency as $f_0 + mask_width$ (mask_width in mel scale)
- define masking range as $[f_0, f_0 + mask_width)$ **in the mel scale**
- transform the masking range back to linear scale