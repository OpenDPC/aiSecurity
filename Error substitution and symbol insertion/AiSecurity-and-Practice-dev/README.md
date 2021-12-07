## Solve Discrimination
- This code is a reproduction of the FS framework. For some special reasons, some results are slightly different from the original journal.
- **Citation:** _Zhang T ,  Zhu T ,  Li J , et al. Fairness in Semi-supervised Learning: Unlabeled Data Help to Reduce Discrimination[J]. IEEE Transactions on Knowledge and Data Engineering, 2020._

## Adversarial Examples
- This code is a reproduction of the following article. For some special reasons, some results are slightly different from the original journal.
- **Citation:** _Kurakin A ,  Goodfellow I ,  Bengio S . Adversarial examples in the physical world[J].  2016._
- **Dataset:** The validation part of ImageNet Large Scale Visual Recognition Challenge (ILSVRC) Dataset, [click here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data?select=imagenet_object_localization_patched2019.tar.gz) to download.
### Something about our work

- Limited by the file size, we provide the supports folder at [Baidu Netdisk](https://pan.baidu.com/s/10wk9auH_Sp9Y-fzN3T_AuQ), with extraction code: 1vkg. We also provide all the images data we used and generated at [Baidu Netdisk](https://pan.baidu.com/s/1xAiLb0zgWwJufZ94r0gcBA), with extraction code: fnbu, as well as our results in pickle format at [Baidu Netdisk]( https://pan.baidu.com/s/1uhd4GdsL51Ja1xRWGKBGWA), with extraction code: fedt.

- Our work is based on _pytorch_. The model for prediction is Inception v3, with pre-trained parameter from torchvision. View *model.MODEL* and *model.predict* for more details.

- We use [_torchattacks_](https://github.com/Harry24k/adversarial-attacks-pytorch) to implement FGSM, BIM and LL, view *model.adv_img_gen* for more details.

- We use _opencv-python_ and _numpy_ to load, transform images. _Opencv_ works well with TensorFlow, whereas it doesn't work as well as _PIL_(pillow) with _torch_. Here are some tips:
  - Opencv use BGR as its default color standard, use
  
    ```python
    import cv2
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ```
  
    to transform color standard of image to RGB.
  
  - *torch.Tensor* structed as Tensor(idx, depth, width, height), whereas opencv returns image as numpy.array(width, height, depth). Use
  
    ```python
    import torch
    torch.tensor(np.transpose(img / 255, (2, 0, 1)), dtype=torch.float)
    ```
  
    to transform numpy to torch.tensor.
  
  - It's necessary to aware the scale of pixels, either between [0, 255] or [0, 1], for some method in *opencv* and *PIL* is stricted on the scale.
  
  - Last but not the least, **take more time on the design of file and variable storage structure and namimg rules** is very important, since the number of files and variables are enormous.
