#pip install git+https://github.com/HYLcool/simhash-py.git
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple git+https://gitee.com/funtianchi/simhash-py.git

# for training
cd lm-training
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
cd ..

# for evaluation
cd lm-evaluation-harness
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -e .
cd ..


# for data-juicer
#git clone https://github.com/alibaba/data-juicer
git clone https://gitee.com/funtianchi/data-juicer
cd data-juicer
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -v -e .[all]

