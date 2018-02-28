CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --resume ../result-wh/nodnet/2017_09_14/0/net/100.ckpt --test 2 --save-freq 1 --lr 0.00001

