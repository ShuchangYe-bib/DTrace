# DTrace
[TMM] DTrace: Dynamic Traceback Learning for Medical Report Generation



## Quick Start

First, clone this repository to your local machine and install the dependencies. 

```bash
git clone git@github.com:ShuchangYe-bib/DTrace.git
cd DTrace
conda create --name dtrace python=3.11
conda activate dtrace
pip install -r requirements.txt
python -m nltk.downloader punkt_tab
```

Now, train and try the model with just few lines of code:

```bash
python3 train.py
python3 inference.py
```
