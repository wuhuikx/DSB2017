#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 1 --lr 0.01
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --resume ../result-wh/nodnet/2017_09_14/0/net/100.ckpt --test 2 --subset 0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 5 --lr 0.01
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --test 2 --subset 5 --lr 0.01 --resume /home/wuhui/3dcnn/result-wh/nodnet/input96/5/net/100.ckpt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 7 --lr 0.01
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --test 2 --subset 7 --lr 0.01 --resume /home/wuhui/3dcnn/result-wh/nodnet/input96/7/net/100.ckpt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 8 --lr 0.01
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --test 2 --subset 8 --lr 0.01 --resume /home/wuhui/3dcnn/result-wh/nodnet/input96/8/net/100.ckpt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 9 --lr 0.01
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --test 2 --subset 9 --lr 0.01 --resume /home/wuhui/3dcnn/result-wh/nodnet/input96/9/net/100.ckpt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 0 --test 2 --resume /home/wuhui/3dcnn/result-wh/nodnet/2017_10_10/0/net/100.ckpt
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --resume ../result-wh/nodnet/2017_09_14/4/net/100.ckpt --test 2 --subset 4
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --save-freq 1 --subset 4 --lr 0.01
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 python main.py --model res18 -b 4 --resume ../result-wh/nodnet/2017_09_14/4/net/100.ckpt --test 2 --subset 4

