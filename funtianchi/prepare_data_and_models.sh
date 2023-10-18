
if [ ! -d "data" ]; then
    echo "make new directory data"
    mkdir data
else
    echo "data directory exists"
fi

cd data

# prepare evaluation data
echo "Preparing evaluation data..."
funget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/eval_data/challenge-data.tar.gz --multi
tar zxvf challenge-data.tar.gz

# prepare base models
echo "Preparing base models..."
mkdir -p models
cd models
echo "Baichuan2-7B"
funget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/huggingface/Baichuan2-7B-Base.tar.gz --multi --block_size=200 --worker=30
tar -zxvf Baichuan2-7B-Base.tar.gz

echo "Falcon-1B"
funget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/huggingface/falcon-rw-1b.tar.gz --multi --block_size=200 --worker=30
tar -zxvf falcon-rw-1b.tar.gz

cd ..

# prepare raw data
echo "Preparing raw datasets..."
mkdir -p raw_data
cd raw_data
funget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/raw_data/raw_data_en.jsonl --multi --block_size=200 --worker=30
funget http://dail-wlcb.oss-cn-wulanchabu.aliyuncs.com/dj-competition/raw_data/raw_data_zh.jsonl --multi --block_size=200 --worker=30

cd -

