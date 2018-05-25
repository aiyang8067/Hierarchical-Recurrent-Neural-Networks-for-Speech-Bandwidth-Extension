import numpy as np
import librosa
import random
import os
import glob

__RAND_SEED = 123
def __fixed_shuffle(inp_list):
    if isinstance(inp_list, list):
        random.seed(__RAND_SEED)
        random.shuffle(inp_list)
        return
    if isinstance(inp_list, np.ndarray):
        np.random.seed(__RAND_SEED)
        np.random.shuffle(inp_list)
        return

    raise ValueError("inp_list is neither a list nor a numpy.ndarray but a "+type(inp_list))

def clip_times(audio, times):

    audio = audio * times
    audio[audio > 1] = 1
    audio[audio < -1] = -1
    return audio


def wav2npy(data_path,save_path,name,fixed_shuffle=True,sample_rate=16000):
	paths = sorted(glob.glob(data_path+"/*.wav"))
	if name=='test':
		fid=open(save_path+'/'+'test_list.scp','w')
		for i in xrange(len(paths)):
			fid.write(paths[i].split('/')[-1]+'\n')
		fid.close()
	if fixed_shuffle:
		__fixed_shuffle(paths)
	for i,path in enumerate(paths):
		audio16k, _ = librosa.load(path, sr=sample_rate, mono=True)
		audio8k = librosa.core.resample(audio16k,sample_rate,sample_rate/2)
		audio8k = librosa.core.resample(audio8k,sample_rate/2,sample_rate)

		if(len(audio8k)==len(audio16k)):
			pass
		elif(len(audio8k)>len(audio16k)):
			audio8k=audio8k[0:len(audio16k)]
		else:
			audio16k=audio16k[0:len(audio8k)]

		audio_up=audio16k-audio8k
		audio_up = clip_times(audio_up, 3)

		if i==0:
			max_len=len(audio_up)
			audio_mat_up=np.array(audio_up,dtype='float32').reshape(1,len(audio_up))
			audio_mat8k=np.array(audio8k,dtype='float32').reshape(1,len(audio8k))
			mask=np.ones(audio_mat_up.shape,dtype='float32')
		else:
			current_len=len(audio_up)
			if current_len>max_len:
				audio_mat_up=np.pad(audio_mat_up,[[0,0],[0,current_len-max_len]],'constant')
				audio_mat_up=np.concatenate((audio_mat_up,np.array(audio_up,dtype='float32').reshape(1,current_len)),axis=0)
				audio_mat8k=np.pad(audio_mat8k,[[0,0],[0,current_len-max_len]],'constant')
				audio_mat8k=np.concatenate((audio_mat8k,np.array(audio8k,dtype='float32').reshape(1,current_len)),axis=0)
				mask=np.pad(mask,[[0,0],[0,current_len-max_len]],'constant')
				mask=np.concatenate((mask,np.ones((1,current_len),dtype='float32')),axis=0)
				max_len=current_len
			else:
				audio_mat_up=np.concatenate((audio_mat_up,np.pad(np.array(audio_up,dtype='float32').reshape(1,current_len),[[0,0],[0,max_len-current_len]],'constant')),axis=0)
				audio_mat8k=np.concatenate((audio_mat8k,np.pad(np.array(audio8k,dtype='float32').reshape(1,current_len),[[0,0],[0,max_len-current_len]],'constant')),axis=0)
				mask=np.concatenate((mask,np.pad(np.ones((1,current_len),dtype='float32'),[[0,0],[0,max_len-current_len]],'constant')),axis=0)

	np.save(save_path+'/'+'TIMIT_'+name+'_up.npy', audio_mat_up)
	np.save(save_path+'/'+'TIMIT_'+name+'_8k.npy', audio_mat8k)
	np.save(save_path+'/'+'TIMIT_'+name+'_mask.npy', mask)

	print name+' data storage is complete!'


wav2npy('datasets/TIMIT/train','datasets/TIMIT','train',fixed_shuffle=True,sample_rate=16000)
wav2npy('datasets/TIMIT/valid','datasets/TIMIT','valid',fixed_shuffle=True,sample_rate=16000)
wav2npy('datasets/TIMIT/test','datasets/TIMIT','test',fixed_shuffle=False,sample_rate=16000)