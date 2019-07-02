Goal: automatically detect pornography, protect sensitive populations.

## I. Intro

Distinguish between sensitive and non-sensitive contents (and body parts too). Previous solutions only classified content to be or not to be blocked. Such a mechanism may compromise user experience -- that's why the new way of masking sensitive parts in the images is proposed.

Problem: object detection task (detecting sensitive regions in the pictures)

YOLO, Faster-RCNN, RetinaNet could be used to locate bounding boxes of sensitive regions -> train dataset with annotated bounding boxes is required.

image-to-image translation approach (translation of image x from sensitive content domain X to y of non-sensitive domain Y). Images from X and Y domains are easy to acquire.

The main issue is finding a big dataset of aligned pairs {x_i, y_i} to train models. That's why the newly proposed method uses training mechanism that uses unpaired training samples to translate image from X to Y.

## II. Background

### A. GANs

Traditional GANs framework is unconditioned -> CGANs (Conditional GANs)

### B. Image-to-image Translation

Translation of image from domain X to domain Y. Examples: image colorization, edge detection, sketch -> photo, image prediction from a normal map

pix2pix, based on CGANs

It is still uncommon to find datasets containing large quantities of paired images.

cycleGAN -> adapts pix2pix to the unpaired settings (style transfer, object transfiguration, season transfer, photo enhancement)

Related work: "Unsupervised Image to Image Translation with Generative Adversarial Networks" [[paper](https://arxiv.org/pdf/1701.02676.pdf)] [[code](https://github.com/zsdonghao/Unsup-Im2Im)]

## III. Proposed Approach

<inspiration from CycleGANs>

### A. Dataset

Images representing both domains were scrapped from Internet (nude women and women wearing bikinis -> filtering to keep only one person on every picture). For X domain (wearing bikinis) 1044 images were selected for Train and 117 images for Test. For Y domain (nude women) 921 images for Train and 103 for Test.

NSFW dataset from related papers: [[pornography classification](https://sites.google.com/site/nudedetection/)], [[pornography video classification](https://sites.google.com/site/pornographydatabase/)]

### B. Loss Function

LSGANs loss was adopted. Translation should be "cycle-consistent". Forward cycle consistency: x -> G(x) -> F(G(x)) ≈ x. Backward cycle consistency: y -> F(y) -> G(F(y)) ≈ y.

### C. Network Architecture

- N-Layers D
- 9-Blocks ResNet G
- U-Net 256 G

## IV. Experiments

256x256 resolution was used. Background was removed to improve quality and speed up the learning process. To segment people in all images the Mask R-CNN was used (sota for semantic and instance segmentation)
