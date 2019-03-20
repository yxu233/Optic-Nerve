Optic nerve analysis to generate final images:

1) Create_masks_Barbara.ijm ==> makes masks from segmented ground truth images

2) Create_control_masks_Barbara.ijm ==> can convert the raw control images (blank ones) to the CORRECT COLOR + makes EMPTY MASKS



Image>>Stacks>>3D project

***maybe need to convert masks from RGB to uint8???


***can add the Masks as a separate channel!!! then do 3D projection!!! then can turn on/off channels with channels tool!!!



1) Import as sequence the raw MASKS (output) && the raw original input images
2) Convert Mask to type uint-8
3) Split channels of raw image
4) Merge channels, setting Masks in the red channel
5) Image>>Stacks>>3D project



***Train with enhanced/rotated ect... images
***get rid of "migrating" blobs
