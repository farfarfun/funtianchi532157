# 更新驱动
#apt-get -y install cuda-drivers

# 安装conda
# mkdir -p ~/miniconda3
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
# bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
# rm -rf ~/miniconda3/miniconda.sh
# ~/miniconda3/bin/conda init bash
# ~/miniconda3/bin/conda init zsh


yes | conda create -n dj_comp python=3.10 --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda activate dj_comp