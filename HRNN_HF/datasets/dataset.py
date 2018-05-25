"""
RNN Vocal Generation Model

TIMIT data feeders.
"""

import numpy as np
import random
import time
import os
import glob

__base = [
    ('Local', 'datasets/'),  
]

__TIMIT_file = 'TIMIT/TIMIT_{}.npy'

__train_mask = lambda s: s.format('train_mask')
__train_up = lambda s: s.format('train_up')
__train8k = lambda s: s.format('train_8k')
__valid_mask = lambda s: s.format('valid_mask')
__valid_up = lambda s: s.format('valid_up')
__valid8k = lambda s: s.format('valid_8k')
__test_mask = lambda s: s.format('test_mask')
__test_up = lambda s: s.format('test_up')
__test8k = lambda s: s.format('test_8k')

def find_dataset(filename):
    for (k, v) in __base:
        tmp_path = os.path.join(v, filename)
        if os.path.exists(tmp_path):
            return tmp_path
    raise Exception('{} NOT FOUND!'.format(filename))

### Basic utils ###
def __round_to(x, y):
    """round x up to the nearest y"""
    return int(np.ceil(x / float(y))) * y

def __normalize(data):
    """To range [0., 1.]"""
    data -= data.min(axis=1)[:, None]
    data /= data.max(axis=1)[:, None]
    return data

def __linear_quantize(data, q_levels):
    """
    floats in (0, 1) to ints in [0, q_levels-1]
    scales normalized across axis 1
    """
    # Normalization is on mini-batch not whole file
    #eps = numpy.float64(1e-5)
    #data -= data.min(axis=1)[:, None]
    #data *= ((q_levels - eps) / data.max(axis=1)[:, None])
    #data += eps/2
    #data = data.astype('int32')

    eps = np.float64(1e-5)
    data *= (q_levels - eps)
    data += eps/2
    data = data.astype('int32')
    return data

def linear2mu(x, mu=255):
    """
    From Joao
    x should be normalized between -1 and 1
    Converts an array according to mu-law and discretizes it

    Note:
        mu2linear(linear2mu(x)) != x
        Because we are compressing to 8 bits here.
        They will sound pretty much the same, though.

    :usage:
        >>> bitrate, samples = scipy.io.wavfile.read('orig.wav')
        >>> norm = __normalize(samples)[None, :]  # It takes 2D as inp
        >>> mu_encoded = linear2mu(2.*norm-1.)  # From [0, 1] to [-1, 1]
        >>> print mu_encoded.min(), mu_encoded.max(), mu_encoded.dtype
        0, 255, dtype('int16')
        >>> mu_decoded = mu2linear(mu_encoded)  # Back to linear
        >>> print mu_decoded.min(), mu_decoded.max(), mu_decoded.dtype
        -1, 0.9574371, dtype('float32')
    """
    x_mu = np.sign(x) * np.log(1 + mu*np.abs(x))/np.log(1 + mu)
    return ((x_mu + 1)/2 * mu).astype('int16')

def mu2linear(x, mu=255):
    """
    From Joao with modifications
    Converts an integer array from mu to linear

    For important notes and usage see: linear2mu
    """
    mu = float(mu)
    x = x.astype('float32')
    y = 2. * (x - (mu+1.)/2.) / (mu+1.)
    return np.sign(y) * (1./mu) * ((1. + mu)**np.abs(y) - 1.)

def __mu_law_quantize(data):
    return linear2mu(data)

def __batch_quantize(data, q_levels, q_type):
    """
    One of 'linear', 'a-law', 'mu-law' for q_type.
    """
    data = data.astype('float64')
    #data = __normalize(data)
    if q_type == 'linear':
        return __linear_quantize(data, q_levels)
    if q_type == 'mu-law':
        # from [0, 1] to [-1, 1]
        #data = 2.*data-1.
        # Automatically quantized to 256 bins.
        return __mu_law_quantize(data)
    raise NotImplementedError

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

def __make_random_batches(inp_list, batch_size,shuffle=True):
    batches = []
    for i in xrange(len(inp_list) / batch_size+1):
        if i==len(inp_list) / batch_size:
            if len(inp_list)%batch_size==0:
                break
            else:
                batches.append(inp_list[i*batch_size:])
        else:
            batches.append(inp_list[i*batch_size:(i+1)*batch_size])

    if shuffle:
        __fixed_shuffle(batches)
    return batches

def __mask_sort(mask_matrix):
    ind=[]
    for i in xrange(len(mask_matrix)):
        ind.append(len(np.where(mask_matrix[i]==1)[0]))
    b=zip(ind,range(len(ind)))
    b.sort(key=lambda x:x[0],reverse=True)
    index=[x[1] for x in b]

    return index

### TIMIT DATASET LOADER ###
def __TIMIT_feed_epoch(files,
                       mask_files, 
                       shuffle,
                       is_train,
                       batch_size,
                       seq_len,
                       overlap,
                       q_levels,
                       q_zero,
                       q_type,
                       real_valued=False):
    """
    Helper function to load blizzard dataset.
    Generator that yields training inputs (subbatch, reset). `subbatch` contains
    quantized audio data; `reset` is a boolean indicating the start of a new
    sequence (i.e. you should reset h0 whenever `reset` is True).

    Feeds subsequences which overlap by a specified amount, so that the model
    can always have target for every input in a given subsequence.

    Assumes all flac files have the same length.

    returns: (subbatch, reset)
    subbatch.shape: (BATCH_SIZE, SEQ_LEN + OVERLAP)
    reset: True or False
    """
    if is_train:
        sort_index=__mask_sort(mask_files)
        batches_8k = __make_random_batches(files[0][sort_index], batch_size,shuffle)
        batches_up = __make_random_batches(files[1][sort_index], batch_size,shuffle)
        mask_batches=__make_random_batches(mask_files[sort_index],batch_size,shuffle)
    else:
        batches_8k = __make_random_batches(files[0], batch_size,shuffle)
        batches_up = __make_random_batches(files[1], batch_size,shuffle)
        mask_batches=__make_random_batches(mask_files,batch_size,shuffle)

    for index,bch_8k in enumerate(batches_8k):

        batch_num=len(bch_8k)
        bch_up=batches_up[index]
        mask=mask_batches[index]
        mask_sum=np.sum(mask,axis=0)
        mask_all0_index=np.where(mask_sum==0)[0]
        if len(mask_all0_index!=0):
            bch_up=bch_up[:,:-len(mask_all0_index)]
            bch_8k=bch_8k[:,:-len(mask_all0_index)]
            mask=mask[:,:-len(mask_all0_index)]

        batch_seq_len = len(bch_8k[0]) 
        batch_seq_len = __round_to(batch_seq_len, seq_len)

        batch_8k = np.zeros(
            (batch_num, batch_seq_len),
            dtype='float64'
        )
        batch_up = np.zeros(
            (batch_num, batch_seq_len),
            dtype='float64'
        )

        mask=np.pad(mask,[[0,0],[0,batch_seq_len-mask.shape[1]]],'constant')
        for i, data in enumerate(bch_8k):
            batch_8k[i, :len(data)] = data
        for i, data in enumerate(bch_up):
            batch_up[i, :len(data)] = data

        batch_8k_real=np.concatenate([
                batch_8k,
                np.full((batch_num, overlap), 0, dtype='float32')
                ], axis=1)
        if not real_valued:
            batch_8k = __batch_quantize(batch_8k, q_levels, q_type)
            batch_up = __batch_quantize(batch_up, q_levels, q_type)

            batch_8k = np.concatenate([
                batch_8k,
                np.full((batch_num, overlap), q_zero, dtype='int32')
                ], axis=1)

            batch_up = np.concatenate([
                batch_up,
                np.full((batch_num, overlap), q_zero, dtype='int32')
                ], axis=1)

        mask = np.concatenate([
            mask,
            np.full((batch_num, overlap), 0, dtype='float32')
        ], axis=1)


        for i in xrange(batch_seq_len // seq_len):
            reset = np.int32(i==0)
            end_flag=np.int32(i==batch_seq_len // seq_len-1)
            subbatch_8k_real=batch_8k_real[:, i*seq_len : (i+1)*seq_len+overlap]
            subbatch_8k = batch_8k[:, i*seq_len : (i+1)*seq_len+overlap]
            subbatch_up = batch_up[:, i*seq_len : (i+1)*seq_len+overlap]
            submask = mask[:, i*seq_len : (i+1)*seq_len+overlap]
            yield (subbatch_8k, subbatch_up,reset, end_flag,submask,batch_num,subbatch_8k_real)

def TIMIT_train_feed_epoch(*args):
    """
    :parameters:
        batch_size: int
        seq_len:
        overlap:
        q_levels:
        q_zero:
        q_type: One the following 'linear', 'a-law', or 'mu-law'

    THE NEW SEG IS:
    20.48hrs 36*256
    3*256
    3*256

    :returns:
        A generator yielding (subbatch, reset, submask)
    """
    # Just check if valid/test sets are also available. If not, raise.
    find_dataset(__valid_up(__TIMIT_file))
    find_dataset(__valid8k(__TIMIT_file))
    find_dataset(__valid_mask(__TIMIT_file))
    find_dataset(__test_up(__TIMIT_file))
    find_dataset(__test8k(__TIMIT_file))
    find_dataset(__test_mask(__TIMIT_file))
    # Load train set
    data_path_8k = find_dataset(__train8k(__TIMIT_file))
    data_path_up = find_dataset(__train_up(__TIMIT_file))
    data_mask_path=find_dataset(__train_mask(__TIMIT_file))
    files=[]
    files.append(np.load(data_path_8k))
    files.append(np.load(data_path_up))
    mask_files=np.load(data_mask_path)
    shuffle=True
    is_train=True
    generator = __TIMIT_feed_epoch(files, mask_files,shuffle,is_train,*args)
    return generator

def TIMIT_valid_feed_epoch(*args):
    """
    See:
        TIMIT_train_feed_epoch
    """
    data_path_8k = find_dataset(__valid8k(__TIMIT_file))
    data_path_up = find_dataset(__valid_up(__TIMIT_file))
    data_mask_path=find_dataset(__valid_mask(__TIMIT_file))
    files=[]
    files.append(np.load(data_path_8k))
    files.append(np.load(data_path_up))
    mask_files=np.load(data_mask_path)
    shuffle=True
    is_train=False
    generator = __TIMIT_feed_epoch(files, mask_files,shuffle,is_train,*args)
    return generator

def TIMIT_test_feed_epoch(*args):
    """
    See:
        TIMIT_train_feed_epoch
    """
    data_path_8k = find_dataset(__test8k(__TIMIT_file))
    data_path_up = find_dataset(__test_up(__TIMIT_file))
    data_mask_path=find_dataset(__test_mask(__TIMIT_file))
    files=[]
    files.append(np.load(data_path_8k))
    files.append(np.load(data_path_up))
    mask_files=np.load(data_mask_path)
    shuffle=False
    is_train=False
    generator = __TIMIT_feed_epoch(files, mask_files,shuffle,is_train,*args)
    return generator
