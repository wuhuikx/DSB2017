python adapt_ckpt.py --model1  net_detector_3 --model2  net_classifier_3  --resume ../result-wh/nodnet/2017_09_06/net/084.ckpt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model1  net_detector_3 --model2  net_classifier_3 -b 1 -b2 1 --save-dir ../result-wh/casenet/2017_09_11 --resume ./result-wh/casenet/start.ckpt --start-epoch 30 --epochs 130
