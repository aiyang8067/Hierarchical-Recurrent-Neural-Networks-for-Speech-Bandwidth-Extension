import numpy as np
import librosa
import random
import os
import glob
import math

__RAND_SEED = 123
def ReadFloatRawMat(datafile,column):
	data = np.fromfile(datafile,dtype=np.float32)
	if len(data)%column!=0:
		print 'ReadFloatRawMat %s, column wrong!'%datafile
		exit()
	if len(data)==0:
		print 'empty file: %s'%datafile
		exit()
	data.shape = [len(data)/column,column]
	return np.float32(data)

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

def wav2npy(data_path,con_data_path,save_path,name,fixed_shuffle=True,sample_rate=16000,frame_len=160,con_dim=100):
	paths = sorted(glob.glob(data_path+"/*.wav"))
	if name=='test':
		fid=open(save_path+'/'+'test_list.scp','w')
		for i in xrange(len(paths)):
			fid.write(paths[i].split('/')[-1]+'\n')
		fid.close()
	con_paths=sorted(glob.glob(con_data_path+"/*.dat"))
	if fixed_shuffle:
		__fixed_shuffle(paths)
		__fixed_shuffle(con_paths)
	for i,path in enumerate(paths):
		print i
		print path
		print con_paths[i]
		audio16k, _ = librosa.load(path, sr=sample_rate, mono=True)
		audio8k = librosa.core.resample(audio16k,sample_rate,sample_rate/2)
		audio8k = librosa.core.resample(audio8k,sample_rate/2,sample_rate)
		condition=ReadFloatRawMat(con_paths[i],1).reshape(1,-1)

		if(len(audio8k)==len(audio16k)):
			pass
		elif(len(audio8k)>len(audio16k)):
			audio8k=audio8k[0:len(audio16k)]
		else:
			audio16k=audio16k[0:len(audio8k)]

		audio_up=audio16k-audio8k
		audio_up = clip_times(audio_up, 3)

		if len(audio8k)>condition.shape[1]/con_dim*frame_len:
			diff=len(audio8k)-condition.shape[1]/con_dim*frame_len
			audio8k=audio8k[:-diff]
			audio_up=audio_up[:-diff]
		elif len(audio8k)<condition.shape[1]/con_dim*frame_len:
			diff=condition.shape[1]/con_dim*frame_len-len(audio8k)
			audio8k=audio8k[:-(int(math.ceil(float(diff)/frame_len))*frame_len-diff)]
			audio_up=audio_up[:-(int(math.ceil(float(diff)/frame_len))*frame_len-diff)]
			condition=condition[:,:-int(math.ceil(float(diff)/frame_len))*con_dim]
		else:
			pass

		if i==0:
			max_len=len(audio_up)
			max_con_len=condition.shape[1]
			audio_mat_up=np.array(audio_up,dtype='float32').reshape(1,len(audio_up))
			audio_mat8k=np.array(audio8k,dtype='float32').reshape(1,len(audio8k))
			mask=np.ones(audio_mat_up.shape,dtype='float32')
			con_mat=condition
		else:
			current_len=len(audio_up)
			current_con_len=condition.shape[1]
			if current_con_len>max_con_len:
				con_mat=np.pad(con_mat,[[0,0],[0,current_con_len-max_con_len]],'constant')
				con_mat=np.concatenate((con_mat,condition),axis=0)
				max_con_len=current_con_len
			else:
				con_mat=np.concatenate((con_mat,np.pad(condition,[[0,0],[0,max_con_len-current_con_len]],'constant')),axis=0)
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
	np.save(save_path+'/'+'TIMIT_'+name+'_con.npy', con_mat)

	print name+' data storage is complete!'


wav2npy('datasets/TIMIT/waveform/train','datasets/TIMIT/bn_norm_condition/train','datasets/TIMIT','train',fixed_shuffle=True,sample_rate=16000)
wav2npy('datasets/TIMIT/waveform/valid','datasets/TIMIT/bn_norm_condition/valid','datasets/TIMIT','valid',fixed_shuffle=True,sample_rate=16000)
wav2npy('datasets/TIMIT/waveform/test','datasets/TIMIT/bn_norm_condition/test','datasets/TIMIT','test',fixed_shuffle=False,sample_rate=16000)