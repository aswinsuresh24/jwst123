import os
from pathlib import Path
import subprocess

overlap_file = 'overlap'
alignment = 'alignment_dispersion.py'
outdir = 'alignment_output_auto'

success_file = 'successful_alignments.txt'
fail_file = 'failed_alignments.txt'

os.makedirs(outdir, exist_ok = True)

with open(success_file, 'w') as success, open(fail_file, 'w') as failed:
    
    with open(overlap_file) as file:

        for line in file:

            if 'Overlap maximized' not in line:
                continue
            if 'F560W' not in line and 'F770W' not in line:
                continue 

            align_image = line.split('MIRI image: ')[1].split(', Reference image: ')[0]
            # align_image = "../" + align_image
            ref_image = line.split(', Reference image: ')[1].split(', Max overlap area')[0]
            # ref_image = "../" + ref_image

            align_name = Path(align_image).stem
            ref_name = Path(ref_image).stem      ## will avoid printing the entire directory in file name 

            pair_outdir = os.path.join(outdir, f'{align_name}_aligned_to_{ref_name}')

            command = ['python', alignment, '--ref', ref_image, '--align', align_image, 
                       '--outdir', pair_outdir, '--plot', '--verbose']
            
            result = subprocess.run(command)

            if result.returncode == 0:
                success.write(f'MIRI image: {align_image}, \nReference image: {ref_image}\n\n')
                success.flush()

            else:
                failed.write(f'MIRI image: {align_image}, \nReference image: {ref_image}\n\n')
                failed.flush()

print('Done')
print(f'Successful pairs written to {success_file}')
print(f'Failed pairs written to {fail_file}')
