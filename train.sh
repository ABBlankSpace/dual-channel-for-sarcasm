CUDA_VISIBLE_DEVICES=0 python main_sarcasm.py --voc_size 30000 --data_dir ./debatev2/spacy/ --name_dataset debatev2 --breakpoint -1 --batch_size 32 --per_checkpoint 32 --n_class 2 --embed_dropout_rate 0.5 --name_model dualbilstm --max_length_sen 100 --n_layers 3