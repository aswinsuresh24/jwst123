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
    parser.add_argument('--datadir', type=str, help='Directory with object data', required=True)
    parser.add_argument('--symlinkdir', type=str, help='Directory to create symlinks', required=True)
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

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    # Create symlinks
    files = glob.glob(os.path.join(args.datadir, '**', '*.fits'), recursive = True)
    print(f'Creating symlinks for {len(files)} files')
    for file in files:
        fl_dst = os.path.join(args.symlinkdir, 'raw', os.path.basename(file))
        create_symlink(file, fl_dst)