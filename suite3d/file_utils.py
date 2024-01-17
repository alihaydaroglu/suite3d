

from os.path import join
from os.path import exists
import os

import numpy as n
from . import tiff_utils as tfu

def find_exp(subjects_dir, subject, date, expnum):
    
    si_params = {}
    
    if not hasattr(expnum, '__iter__'):
        expnum = [expnum]
    tif_paths = []
    for expn in expnum: 
        exp_dir = find_expt_file((subject,date,expn), 'root', dirs=[subjects_dir])
        tif_paths += tfu.get_tif_paths(exp_dir)

    exp_str = "%s_%s_" % (subject,date)
    for expn in expnum: exp_str += '%d-' % expn
    exp_str = exp_str[:-1]
    si_params['rois'] = tfu.get_meso_rois(tif_paths[0])
    si_params['vol_rate'] = tfu.get_vol_rate(tif_paths[0])
    si_params['line_freq'] = 2 * tfu.get_tif_tag(tif_paths[0],'SI.hScan2D.scannerFrequency', number=True)
    
    return tif_paths, si_params, exp_str
def get_si_params(tif_path):
    si_params = {}
    si_params['rois'] = tfu.get_meso_rois(tif_path)
    si_params['vol_rate'] = tfu.get_vol_rate(tif_path)
    si_params['line_freq'] = 2 * tfu.get_tif_tag(tif_path,'SI.hScan2D.scannerFrequency', number=True)
    return si_params


def find_expt_file(expt_info,file, dirs = None, verbose = False):
    
    subject, expt_date, expt_num = expt_info
       
    file_names = {'timeline' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'Timeline.mat'])),
                 'protocol' : join(subject, expt_date, str(expt_num), 
                                  'Protocol.mat'),
                 'block' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'Block.mat'])),
                 'suite2p' : join(subject, expt_date, 'suite2p'),
                 'facemap' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'eye_proc.npy'])),
                 'eye_log' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'eye.mat'])),
                 'eye_video' : join(subject, expt_date, str(expt_num),
                                    '_'.join([expt_date,str(expt_num),subject,
                                              'eye.mj2'])),
                 'root' : join(subject,expt_date,str(expt_num)),
                 'date' : join(subject,expt_date),
                 'pfile': join(subject, expt_date, str(expt_num),
                                    '_'.join([subject,expt_date,str(expt_num)
                                              ]) + '.p'),
                 'subject' : subject}
    
                              
    file_name = file_names.get(file.lower(), 'invalid')
    # print(file_name)
    if file_name == 'invalid':
        print('File type is invalid. Valid file types are ' 
              + str(list(file_names.keys())))
        return

    if dirs is None:            
        assert False
    # print(file_name)
    for d in dirs:
        if verbose:
             print("Looking for %s in %s" % (file_name, d))
        # if verbose: print("Looking for: ", print(os.path.join(d,file_name)))F
        if exists(join(d,file_name)):
            file_path = join(d,file_name)
            if verbose: print("Found")
            # print(file_path)
            break
    
    if 'file_path' in locals():
        return file_path
    else: 
        print('File could not be found! Be sure that ' + 
              'cortex_lab_utils.expt_dirs() includes all valid directories.')