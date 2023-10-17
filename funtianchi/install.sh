pip install git+https://gitee.com/funtianchi/simhash-py.git

# for training
cd lm-training
pip install -r requirements.txt
cd ..

# for evaluation
cd lm-evaluation-harness
pip install -e .
cd ..


# for data-juicer
git clone https://gitee.com/funtianchi/data-juicer
cd data-juicer
pip install -v -e .[all]

