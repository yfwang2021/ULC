# prepare for the baseline FSIW 
python  ./src/main.py \
--method FSIW --mode pretrain --model_ckpt_path ./models/fsiw0_cd_5 \
--data_path ./data/data.txt \
--data_cache_path ./data \
--fsiw_pretraining_type fsiw0 --CD 5

python  ./src/main.py \
--method FSIW --mode pretrain --model_ckpt_path ./models/fsiw1_cd_5 \
--data_path ./data/data.txt \
--data_cache_path ./data \
--fsiw_pretraining_type fsiw1 --CD 5

# MLP
for k in 0.00001
do
    for j in 0.0001
    do
        for i in 0 1 2 3 4
        do
            python ./src/main.py --method ULC \
            --mode train --l2_reg $k --cuda_device 1 --lr $j --CD 7 --batch_size 1024 --optimizer Adam --seed $i &
        done
        wait
    done
done

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method FSIW --mode train --dataset_source criteo \
    --pretrain_fsiw0_model_ckpt_path ./models/fsiw0_cd_5 \
    --pretrain_fsiw1_model_ckpt_path ./models/fsiw1_cd_5 \
    --l2_reg 0.000001 --cuda_device 1 --lr 0.0005 --seed $i --optimizer Adam &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Vanilla --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Oracle --mode train --dataset_source criteo \
    --l2_reg 0.000001 --cuda_device 1 --lr 0.0005 --seed $i --optimizer Adam &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method DFM --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \ 
    --method nnDF --mode train --dataset_source criteo \
    --l2_reg 0.000001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --CD 5 &
done
wait

# DeepFM
for k in 0.0001
do
    for j in 0.0001
    do
        for i in 0 1 2 3 4
        do
            python ./src/main.py --method ULC \
            --mode train --l2_reg $k --cuda_device 1 --lr $j --CD 7 --batch_size 1024 --optimizer Adam --seed $i --base_model DeepFM &
        done
        wait
    done
done

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method FSIW --mode train --dataset_source criteo \
    --pretrain_fsiw0_model_ckpt_path ./models/fsiw0_cd_5 \
    --pretrain_fsiw1_model_ckpt_path ./models/fsiw1_cd_5 \
    --l2_reg 0.00001 --cuda_device 1 --lr 0.0005 --seed $i --optimizer Adam --base_model DeepFM &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Vanilla --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0005 --seed $i --optimizer Adam --base_model DeepFM &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Oracle --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam --base_model DeepFM &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method DFM --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam --base_model DeepFM &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \ 
    --method nnDF --mode train --dataset_source criteo \
    --l2_reg 0.000001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --CD 5 --base_model DeepFM &
done
wait

# AutoInt
for k in 0.0001
do
    for j in 0.0001
    do
        for i in 0 1 2 3 4
        do
            python ./src/main.py --method ULC \
            --mode train --l2_reg $k --cuda_device 1 --lr $j --CD 7 --batch_size 1024 --optimizer Adam --seed $i --base_model AutoInt &
        done
        wait
    done
done

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method FSIW --mode train --dataset_source criteo \
    --pretrain_fsiw0_model_ckpt_path ./models/fsiw0_cd_5 \
    --pretrain_fsiw1_model_ckpt_path ./models/fsiw1_cd_5 \
    --l2_reg 0.00001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --base_model AutoInt &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Vanilla --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0005 --seed $i --optimizer Adam --base_model AutoInt &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Oracle --mode train --dataset_source criteo \
    --l2_reg 0.00001 --cuda_device 1 --lr 0.0005 --seed $i --optimizer Adam --base_model AutoInt &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method DFM --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam --base_model AutoInt &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \ 
    --method nnDF --mode train --dataset_source criteo \
    --l2_reg 0.000001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --CD 5 --base_model AutoInt &
done
wait

# DCNV2
for k in 0.0001
do
    for j in 0.0001
    do
        for i in 0 1 2 3 4
        do
            python ./src/main.py --method ULC \
            --mode train --l2_reg $k --cuda_device 1 --lr $j --CD 7 --batch_size 1024 --optimizer Adam --seed $i --base_model DCNV2 &
        done
        wait
    done
done

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method FSIW --mode train --dataset_source criteo \
    --pretrain_fsiw0_model_ckpt_path ./models/fsiw0_cd_5 \
    --pretrain_fsiw1_model_ckpt_path ./models/fsiw1_cd_5 \
    --l2_reg 0.00001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --base_model DCNV2 &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Vanilla --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam --base_model DCNV2 &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method Oracle --mode train --dataset_source criteo \
    --l2_reg 0.00001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --base_model DCNV2 &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \
    --method DFM --mode train --dataset_source criteo \
    --l2_reg 0.0001 --cuda_device 1 --lr 0.0001 --seed $i --optimizer Adam --base_model DCNV2 &
done
wait

for i in 0 1 2 3 4
do
    python ./src/main.py \ 
    --method nnDF --mode train --dataset_source criteo \
    --l2_reg 0.000001 --cuda_device 1 --lr 0.001 --seed $i --optimizer Adam --CD 5 --base_model DCNV2 &
done
wait