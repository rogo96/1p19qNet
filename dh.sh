#!/bin/bash

python3 train_model.py --lr=1e-4 --num_epochs=100 --max_r=100 --gpu=0 --n_fold=2 \
	--data_dir=0214_feature --wd=5e-4 --all_data
#python3 test.py --excel 

#sleep 5m
#
#num=1

#while [ ${num} -lt 11 ]
#do
#
#	python3 test_and.py --dname_model_1=0218_reg_1_R50_model/${num} --dname_model_19=0218_reg_19_R50_model/${num} --gpu=0 --mode=Regression --data_dir=0214_feature --loss=one --max_r=50 \
#		--seed_num=${num} --gpu=0 --excel --infer
#
#	num=`expr ${num} + 1 `
#
#done

#sleep 1m
#
#num=1
#
#while [ ${num} -lt 2 ]
#do
#
#        python3 test_and_bootstrap.py --dname_model_1=0214_reg_1_model --dname_model_19=0214_reg_19_model --gpu=0 --mode=Regression --data_dir=0214_feature --loss=one \
#                --seed_num=${num} --gpu=0 --excel
#
#       num=`expr ${num} + 1 `

#done
