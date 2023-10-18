pip install  -r requirements.txt

pip install funpypi funbuild funget funfile
yes | pip uninstall funfile && pip install -U git+https://gitee.com/farfarfun/funfile.git
yes | pip uninstall funget && pip install -U git+https://gitee.com/farfarfun/funget.git


#pip install git+https://github.com/HYLcool/simhash-py.git
pip install  git+https://gitee.com/funtianchi/simhash-py.git

# for training
cd lm-training
pip install  -r requirements.txt
cd ..

# for evaluation
cd lm-evaluation-harness
pip install  -e .
cd ..


# for data-juicer
#git clone https://github.com/alibaba/data-juicer
git clone https://gitee.com/funtianchi/data-juicer
cd data-juicer
pip install  -v -e .[all]

