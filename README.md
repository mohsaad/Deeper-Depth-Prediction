# Deeper-Depth-Prediction

Depth prediction using Deep Residual Networks.

Original code and paper by @iro-cp found here: https://github.com/iro-cp/FCRN-DepthPrediction

I was also helped by @iapatil's version of this, found here:
https://github.com/iapatil/depth-semantic-fully-conv

Written in PyTorch. To run, download the pretrained numpy weights from [here](http://campar.in.tum.de/files/rupprecht/depthpred/NYU_ResNet-UpProj.npy) and save them in the current directory.

Then, activate a PyTorch environment and run

```
python predict.py <color_image>
```

The output will be saved as `output_image.png`.

My post detailing the paper and implementation can be found [here](http://mohsaad.com/2018/03/10/Deeper-Depth-Prediction/).
