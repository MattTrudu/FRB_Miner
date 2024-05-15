from setuptools import setup, find_packages

setup(
    name='FRB_Miner',
    version='0.1',
    packages=find_packages(),
    scripts=['bin/rfi_zapper.py', 'bin/launch_heimdall.py',
             'bin/subband_filterbank.py', 'bin/prepare_for_fetch.py',
             'bin/frb_miner.py','bin/plan_subbands.py', 'bin/plot_data.py',
             'bin/clean_filterbank.py', 'bin/plot_frb_candidate.py'],
    author='Matteo Trudu',
    author_email='matteo.trudu@inaf.it',
    description='FRB searcher and analyser',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MattTrudu/FRB_Miner.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
