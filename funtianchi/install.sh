pip install  -r requirements.txt

pip install funpypi funbuild funget funfile -i https://pypi.tuna.tsinghua.edu.cn/simple
yes | pip uninstall funfile && pip install -U git+https://gitee.com/farfarfun/funfile.git -i https://pypi.tuna.tsinghua.edu.cn/simple
yes | pip uninstall funget && pip install -U git+https://gitee.com/farfarfun/funget.git -i https://pypi.tuna.tsinghua.edu.cn/simple


#pip install git+https://github.com/HYLcool/simhash-py.git
pip install  git+https://gitee.com/funtianchi/simhash-py.git -i https://pypi.tuna.tsinghua.edu.cn/simple

# for training
cd lm-training
pip install  -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..

# for evaluation
cd lm-evaluation-harness
pip install  -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..


# for data-juicer
#git clone https://github.com/alibaba/data-juicer
pip install git+https://github.com/farfarfun/data-juicer.git -i https://pypi.tuna.tsinghua.edu.cn/simple
# git clone https://gitee.com/funtianchi/data-juicer
# cd data-juicer
# pip install  -v -e .[all]

