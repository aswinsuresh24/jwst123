import glob, os
import argparse

def create_parser():
    '''
    Create a parser for the command line arguments

    Returns:
    -------
    parser : argparse.ArgumentParser
        arg parser
    '''
    parser = argparse.ArgumentParser(description='Symlink data for jwst123 run')
    parser.add_argument('--datadir', type=str, help='Full path to directory with object data', required=True)
    parser.add_argument('--symlinkdir', type=str, 
                        help='Full path to directory containing the raw directory, in which symlinks will be created', 
                        required=True)
    parser.add_argument('--proc_dirs', nargs = '*', type=str, help='List of directories that have processed files', 
                        default = [], required=False)
    return parser

def create_symlink(src, dst):
    '''
    Create a symlink from src to dst

    Parameters:
    ----------
    src : str
        source file
    dst : str
        destination file

    Returns:
    -------
    None
    '''
    if not os.path.exists(dst):
        try:
            os.symlink(src, dst)
        except:
            os.unlink(dst)
            os.symlink(src, dst)
    else:
        print(f'{dst} already exists')

def remove_proc_files(files, dir):
    """
    Remove files that have already been processed in dir

    Parameters:
    ----------
    files : list
        list of all files in the data directory
    dir : str
        directory where some files have already been processed

    Returns:
    -------
    new_files : list
        list of files that have not been processed
    """
    proc_files = glob.glob(os.path.join(dir, 'raw', '*.fits'), recursive = True)
    proc_files = [os.path.realpath(i) for i in proc_files]
    new_files = list(set(files) - set(proc_files))

    return new_files

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # Create symlinks
    files = glob.glob(os.path.join(args.datadir, '**', '*.fits'), recursive = True)
    for procdir in args.proc_dirs:
        files = remove_proc_files(files, procdir)
    print(f'Creating symlinks for {len(files)} files')
    for file in files:
        fl_dst = os.path.join(args.symlinkdir, 'raw', os.path.basename(file))
        create_symlink(file, fl_dst)