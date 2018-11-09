import os, sys
from pprint import pprint

sbatch_template = '''#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name="{job_name}"
#
#SBATCH --mail-type=FAIL,REQUEUE,STAGE_OUT,TIME_LIMIT_50
#SBATCH --mail-user={email}

#SBATCH -p gpu
#SBATCH --gres=gpu:1

#SBATCH -t 49:59:59
##SBATCH -p standard

#SBATCH --error="{file_err}.err.log"
#SBATCH --output="{file_out}.out.log"
##SBATCH --nodelist=ai0[1-6]

module load anaconda3

. /apps/software/standard/core/anaconda3/5.2.0/etc/profile.d/conda.sh
conda activate {env_path}

python experiments/test.py {params}
'''

if __name__ == '__main__':

    base_str = '0_'
    sign = '1'
    base_str = base_str + sign

    if not os.path.isdir(base_str):
        os.mkdir(base_str)

    data_name_bases = [
        # 'ga-b3.0,0.4-j100-i100',
        # 'ga-b2.5,0.5-j100-i100',
        # 'ga-b2,1-j100-i100',
        # 'ga-b1.5,1.5-j100-i100',
        # 'ga-b1.25,3.0-j100-i100',
        # 'ga-b1.25,4.0-j100-i100',

        # 'ga-b5,1-j100-i100',
        # 'ga-b2,2-j100-i100',
        # 'ga-b1,1-j100-i100',
        # 'ga-b1,2-j100-i100',
        # 'ga-b1,5-j100-i100',

        # 'be-b5,1-j100-i100',
        # 'be-b16,8-j100-i100',
        # 'be-b8,4-j100-i100',
        # 'be-b4,2-j100-i100',
        # 'be-b2,1-j100-i100',
        # 'be-b8,8-j100-i100',
        # 'be-b4,4-j100-i100',
        # 'be-b2,2-j100-i100',
        # 'be-b1,1-j100-i100',
        # 'be-b1,2-j100-i100',
        # 'be-b2,4-j100-i100',
        # 'be-b4,8-j100-i100',
        # 'be-b8,16-j100-i100',
        # 'be-b1,5-j100-i100',

        # 'ex-b0.1-j100-i100',
        # 'ex-b0.2-j100-i100',
        # 'ex-b0.4-j100-i100',
        # 'ex-b0.8-j100-i100',
        # 'ex-b1.0-j100-i100',
        # 'ex-b1.5-j100-i100',
        # 'ex-b2.0-j100-i100',
        # 'ex-b4.0-j100-i100',
        # 'ex-b6.0-j100-i100',

        # 'ex-b1.0-j5-i5',
        '0.8k1',
        '0.4k1',
        '0.2k1',
        '0.1k1',
    ]

    f = open(os.path.join(base_str, base_str+'.txt'), 'w')
    f.write(' '.join(data_name_bases))
    f.close()

    f = open('.config')
    email, env_path = f.read().split(' ')
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
        }
        sbatch = sbatch_template.format(**kwargs)
        pprint(kwargs)
        f = open('submit.bash', 'w')
        f.write(sbatch)
        f.close()
        os.system('sbatch submit.bash')
        if os.path.isfile('submit.bash'):
            os.remove('submit.bash')
