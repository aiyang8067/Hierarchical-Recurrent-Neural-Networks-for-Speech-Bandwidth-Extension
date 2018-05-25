The HRNN system in the paper: 
* Zhen-Hua Ling , Yang Ai, Yu Gu, and Li-Rong Dai, "Waveform Modeling and Generation Using Hierarchical Recurrent Neural Networks for Speech Bandwidth Extension," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 26, no. 5, pp. 883-894, 2018.
Usage:
First enter the root directory of the folder: `cd HRNN_HF`.

Data preparation:
Put the train, validiation and test waveforms (16kHz sample rate) into the corresponding folder in directory 'datasets/TIMIT',
then run `python datasets/TIMIT/_2npy_hf.py` to generate the packaged data.

Traning and validiation:
Run:
`THEANO_FLAGS='floatX=float32,device=gpu0,allow_gc=False,lib.cnmem=0.95' python -u models/three_tier/three_tier_train_valid.py --exp BEST_3TIER --seq_len 480 --big_frame_size 16 --frame_size 4 --weight_norm True --emb_size 256 --skip_conn False --dim 1024 --n_rnn 1 --rnn_type LSTM --learn_h0 True --q_levels 256 --q_type mu-law --which_set TIMIT --batch_size 64`

Test:
Run:
`THEANO_FLAGS='floatX=float32,device=gpu0,allow_gc=False,lib.cnmem=0.95' python -u models/three_tier/three_tier_test.py --exp BEST_3TIER --seq_len 480 --big_frame_size 16 --frame_size 4 --weight_norm True --emb_size 256 --skip_conn False --dim 1024 --n_rnn 1 --rnn_type LSTM --learn_h0 True --q_levels 256 --q_type mu-law --which_set TIMIT --batch_size 64`