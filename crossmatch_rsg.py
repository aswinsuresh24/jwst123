## python3 crossmatch_rsg.py \
##    --rsg /data/rwisenbaker/jwst_data/RSG_catalog.csv \
##    --photdir /data/rwisenbaker/jwst_data/M82/dolphot \
##    --outdir /data/rwisenbaker/jwst_data/M82/RSG_matches

## path to rsg catalog: /data/rwisenbaker/jwst_data
## path to phot files: /data/rwisenbaker/jwst_data/M*/dolphot

import argparse
from pathlib import Path
import pandas as pd

def main():

    parser = argparse.ArgumentParser(
        description='Crossmatch all dolphot .phot catalogs with an rsg catalog')

    parser.add_argument('--rsg', required=True)
    parser.add_argument('--photdir', required=True)
    parser.add_argument('--outdir', required=True)

    args = parser.parse_args()

    phot_root = Path(args.photdir)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    ## reading rsg catalog (.csv)
    rsg = pd.read_csv(args.rsg)

    #print('RSG catalog columns:')
    #print(list(rsg.columns))

    ## storing rsg coordinates
    rsg_coords = set(zip(rsg['X'], rsg['Y']))

    ## finding all dolphot .phot files recursively
    phot_files = sorted(phot_root.rglob('*.phot'))

    if len(phot_files) == 0:
        print('No .phot files found')
        return

    print(f'Found {len(phot_files)} .phot files')

    total_matches = 0

    for phot_file in phot_files:

        print(f'Processing: {phot_file}')

        ## reading dolphot .phot files
        phot = pd.read_csv(
            phot_file,
            sep=r'\s+',
            comment='#',
            header=None)

        ## dolphot columns: column 3 = X (index 2), column 4 = Y (index 3)
        phot_coords = list(zip(phot.iloc[:, 2], phot.iloc[:, 3]))

        ## keeping only dolphot rows whose X, Y coordinates
        ## appear in the rsg catalog (matches = mask)
        matches = [coord in rsg_coords for coord in phot_coords]

        rsg_matches = phot[matches]

        ## preserving original folder structure
        relative_path = phot_file.relative_to(phot_root)
        output_file = out_root / relative_path

        output_file.parent.mkdir(parents=True, exist_ok=True)

        rsg_matches.to_csv(
            output_file,
            sep=' ',
            header=False,
            index=False)

        print(f'Matches: {len(rsg_matches)}')

        total_matches += len(rsg_matches)

    print('\nDone')
    print(f'Processed catalogs: {len(phot_files)}')
    print(f'Total RSG matches: {total_matches}')


if __name__ == '__main__':
    main()
