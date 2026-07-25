import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import glob, os
import itertools

def get_filters(columns):
    """
    get unique filters and lines from dolphot column files

    Parameters
    ----------
    columns : str
        path to dolphot column file

    Returns
    -------
    lines : list
        list of lines from dolphot column file
    filters : list
        list of unique filters
    """
    with open(columns, 'r') as f:
        lines = f.readlines()
    lines = np.array(lines)

    filters = []
    for line in lines:
        if 'Normalized count rate, NIRCAM' in line:
            filters.append(line.split('NIRCAM_')[-1].split('\n')[0])

    return lines, filters

def map_columns(columns):
    """
    generate dictionaries mapping column names to column indices from dolphot column files

    Parameters
    ----------
    columns : str
        path to dolphot column file

    Returns
    -------
    col_dict : dict
        dictionary mapping column names to column indices for combined photometry
    filt_dict : dict
        dictionary mapping filter names to counts, error and flag column indices for
        individual images
    """
    #combined photometry columns
    column_strings = ['Object X position', 'Object Y position', 'Signal-to-noise', 
                      'Object sharpness', 'Crowding', 'Object type']
    column_key = ['X', 'Y', 'SNR', 'Sharpness', 'Crowding', 'Type']
    
    #get unique filters and lines from column file
    lines, filters = get_filters(columns)

    #define keys and strings for instrumental magnitudes and uncertainties
    mag_strings = [f'Instrumental VEGAMAG magnitude, NIRCAM_{filt}' for filt in filters]
    magerr_strings = [f'Magnitude uncertainty, NIRCAM_{filt}' for filt in filters]
    mag_key, magerr_key = [i + '_mag' for i in filters], [i + '_err' for i in filters]

    keys = column_key+mag_key+magerr_key
    strings = column_strings+mag_strings+magerr_strings
    
    col_dict = {key: [] for key in keys}
    
    #get column indices for combined photometry
    for key, string in zip(keys, strings):
        col = lines[np.char.find(lines, string) > 0]
        if len(col) > 0:
            col_dict[key] = int(col[0].split('.')[0]) - 1
        else:
            col_dict[key] = None

    #get column indices for individual images
    filt_dict = {key: [] for key in filters}
    for filter_ in filters:
        flt_keys = ['Counts', 'Err', 'Flag']
        flt_dict = {key: [] for key in flt_keys}
        idx_cts, idx_err, idx_flag = [], [], []
        for line in lines:
            #get column indices for counts, errors and flags
            #count uncertainty is used to get the index since 'Normalized count rate' is not unique
            if (filter_ in line) & ('Normalized count rate uncertainty' in line):
                idx = int(line.split(' ')[0].split('.')[0]) - 1
                idx_cts.append(str(idx - 1))
                idx_err.append(str(idx))
                idx_flag.append(str(idx + 9))
        #first index corresponds to combined photometry
        flt_dict['Counts'] = idx_cts[1:]
        flt_dict['Err'] = idx_err[1:]
        flt_dict['Flag'] = idx_flag[1:]
        filt_dict[filter_] = flt_dict
    
    return col_dict, filters, filt_dict

def save_photfiles(photfile_path, outdir, obj, chunksize = 100000):
    """
    save photometry files with cuts applied to smaller csv files

    Parameters
    ----------
    photfile_path : str
        path to directory containing dolphot photometry files
    outdir : str
        path to directory to save csv files
    obj : str
        object name
    chunksize : int
        number of rows to read from photometry file at a time

    Returns
    -------
    None
    """
    photfiles = sorted(glob.glob(os.path.join(photfile_path, '*phot')))[:1]
    for i, photfile in enumerate(photfiles):
        column_file = photfile + '.columns'
        #map columns to indices
        col_idx, filters, _ = map_columns(column_file)

        #read photometry file in chunks
        photdf = pd.read_csv(photfile, sep = '\s+', memory_map = True, 
                             header = None, iterator = True, chunksize = chunksize)
        
        for j, _df in tqdm(enumerate(photdf)):
            #apply cuts
            cuts = (_df[col_idx['SNR']] >= 10) & \
                    ((_df[col_idx['Sharpness']])**2 <= 0.01) & \
                    (_df[col_idx['Crowding']] <= 0.5) & \
                    (_df[col_idx['Type']] <=2)
            
            #generate index by combining x and y positions
            #the x, y positions are rounded to the floor of the value since dolphot output 
            #does not use the same star center from different runs
            # df_idx = np.apply_along_axis(lambda x: '_'.join(x), 0, np.stack((_df[2].to_numpy(dtype = str), _df[3].to_numpy(dtype = str))))
            df_idx = ["{:.2f}_{:.2f}".format(i, j) for i, j in zip(np.floor(np.array(_df[2])), np.floor(np.array(_df[3])))]
            _df['idx'] = df_idx
            _df.set_index('idx', inplace = True)
            _df[cuts].to_csv(f'{outdir}/{obj}_{i}_{j}.csv', mode = 'a', header = False)

def create_common_rsg_cat(common_rsg, dfs, columns, outfile):
    """
    save combined photometry for common rsgs to a csv file

    Parameters
    ----------
    common_rsg : list
        list of unique rsgs
    dfs : list
        list of dataframes containing photometry
    columns : list
        list of column files

    Returns
    -------
    None
    """
    combined_data = []
    all_filters = ['F115W', 'F150W', 'F200W', 'F277W', 'F360M']

    for idx in tqdm(common_rsg):
        rsg_dat = []
        posn = []
        for filt in all_filters:
            ct, er = 0, 0
            for df, col in zip(dfs, columns):
                if not idx in df.index:
                    continue
                col_dict, filters, filt_dict = map_columns(col)
                posn.append([df[str(col_dict['X'])].loc[idx], df[str(col_dict['Y'])].loc[idx]])
                if not filt in filters:
                    continue
                #weighted average of magnitudes
                cts, err, flag = np.array(df[filt_dict[filt]['Counts']].loc[idx]), np.array(df[filt_dict[filt]['Err']].loc[idx]), np.array(df[filt_dict[filt]['Flag']].loc[idx])
                ctsc, errc, flagc = cts[(flag < 8) & (cts > 0)], err[(flag<8) & (cts > 0)], flag[(flag < 8) & (cts > 0)]
                ct += np.sum(ctsc/errc**2)
                er += np.sum(1/errc**2)

            if ct > 0:
                mag = -2.5*np.log10(ct/er)
                #dolphot calculates magnitude uncertainties as below
                magerr = 1.0857362*(1/np.sqrt(er))/(ct/er)
            else:
                mag, magerr = 99.99, 99.99
            rsg_dat.extend([mag, magerr])
        rsg_dat = [np.mean(np.array(posn)[:, 0]), np.mean(np.array(posn)[:, 1])] + rsg_dat
        combined_data.append(rsg_dat)

    combined_data = np.array(combined_data)
    #save combined photometry to csv
    df_col = ['X', 'Y'] + [[f'{filt}_mag', f'{filt}_err'] for filt in all_filters]
    df_col = list(itertools.chain(*df_col))
    df = pd.DataFrame(combined_data, columns = df_col)
    df.to_csv(outfile, index = False)

if __name__ == '__main__':
    dfs = []
    for file in sorted(glob.glob('ngc628_phot/rsg*.csv')):
        df = pd.read_csv(file)
        df.set_index('idx', inplace = True)
        dfs.append(df)

    idx = []
    for df in dfs:
        idx.extend(df.index.values)

    common_rsg = np.unique(idx)

    columns = sorted(glob.glob('ngc628_phot/*columns'))
    create_common_rsg_cat(common_rsg, dfs, columns, 'ngc628_combined_phot.csv')