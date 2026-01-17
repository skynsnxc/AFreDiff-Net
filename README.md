# AFreDiff-Net
# PH2 
python train_newdata.py --dataset '../ph2/' --arch AFreDiff-Net --name ph2_nb_testing_batch_8 --img_ext .bmp --mask_ext _lesion.bmp  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8
# ISIC2017 
python train_newdata.py --dataset '../isic2017/' --arch UCM_Net --name isic2017_nb_testing_batch_8 --img_ext .jpg --mask_ext _segmentation.png  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8
# ISIC2018
python train_newdata.py --dataset '../isic2018/' --arch UCM_Net --name isic2018_nb_testing_batch_8 --img_ext .png --mask_ext .png  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8
# Kvasir-SEG
python train_newdata.py --dataset '../Kvasir-SEG/' --arch UCM_Net --name Kvasir-SEG_nb_testing_batch_8 --img_ext .png --mask_ext .png  --epochs 300 --loss GT_BceDiceLoss_new1 --batch_size 8
# PH2
python val.py --name ph2_nb_testing_batch_8
# ISIC2017
python val.py --name isic2017_nb_testing_batch_8
# ISIC2018
python val.py --name isic2018_nb_testing_batch_8
# Kvasir-SEG
python val.py --name isic2018_nb_testing_batch_8
