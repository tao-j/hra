import os, sys
from pprint import pprint

sbatch_template = '''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name="{job_name}"
#
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT,TIME_LIMIT_50
#SBATCH --mail-user={email}

#SBATCH -p intel
##SBATCH --gres=gpu:1

#SBATCH -t 49:59:59
##SBATCH -p standard

#SBATCH --error="{file_err}.err.log"
#SBATCH --output="{file_out}.out.log"
##SBATCH --nodelist=ai0[1-6]

# module load anaconda3

. {conda_sh}

conda activate {env_path}

python experiments/test.py {params}
'''

if __name__ == '__main__':

    base_str = '2_new_data'
    sign = '-1'
    base_str = base_str + sign

    if not os.path.isdir(base_str):
        os.mkdir(base_str)

    data_name_bases = [
        'ga-3.0,0.4-j25-i25',
        'ga-2.5,0.5-j25-i25',
        'ga-2,1-j25-i25',
        'ga-1.5,1.5-j25-i25',
        'ga-1.25,3.0-j25-i25',
        'ga-1.25,4.0-j25-i25',

        'ga-5,1-j100-i100',
        'ga-2,2-j100-i100',
        'ga-1,1-j100-i100',
        'ga-1,2-j100-i100',
        'ga-1,5-j100-i100',

        # 'be-5,1-j100-i100',
        # 'be-16,8-j100-i100',
        # 'be-8,4-j100-i100',
        # 'be-4,2-j100-i100',
        # 'be-2,1-j100-i100',
        # 'be-8,8-j100-i100',
        # 'be-4,4-j100-i100',
        # 'be-2,2-j100-i100',
        # 'be-1,1-j100-i100',
        # 'be-1,2-j100-i100',
        # 'be-2,4-j100-i100',
        # 'be-4,8-j100-i100',
        # 'be-8,16-j100-i100',
        # 'be-1,5-j100-i100',

        # 'ex-0.1-j100-i100',
        # 'ex-0.2-j100-i100',
        # 'ex-0.4-j100-i100',
        # 'ex-0.8-j100-i100',
        # 'ex-1.0-j100-i100',
        # 'ex-1.5-j100-i100',
        # 'ex-2.0-j100-i100',
        # 'ex-4.0-j100-i100',
        # 'ex-6.0-j100-i100',

        # 'ex-1.0-j5-i5',

        # '0.8k1',
        # '0.4k1',
        # '0.2k1',
        # '0.1k1',
    ]

    f = open(os.path.join(base_str, base_str+'.txt'), 'w')
    f.write(' '.join(data_name_bases))
    f.close()

    f = open('.config')
    email, env_path, conda_sh = f.read().split(' ')
    for idx, job_name in enumerate(data_name_bases):
        # params = [base_str, job_name]
        params = [job_name, sign]
        kwargs = {
            'job_name': job_name,
            'file_err': os.path.join(base_str, job_name),
            'file_out': os.path.join(base_str, job_name),
            'params': ' '.join(params),
            'email': email,
            'env_path': env_path,
            'conda_sh': conda_sh
        }
        sbatch = sbatch_template.format(**kwargs)
        pprint(kwargs)
        f = open('submit.bash', 'w')
        f.write(sbatch)
        f.close()
        os.system('sbatch submit.bash')
        if os.path.isfile('submit.bash'):
            os.remove('submit.bash')
