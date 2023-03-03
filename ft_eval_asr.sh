CUDA_VISIBLE_DEVICES=1 python asr_finetuning.py \
    --model_name_or_path EdoAbati/whisper-large-v2-it \
    --dataset_name_or_path RiTA-nlp/italic-easy \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_input_length_in_seconds 15 \
    --gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES=1 python asr_finetuning.py \
    --model_name_or_path EdoAbati/whisper-large-v2-it \
    --dataset_name_or_path RiTA-nlp/italic-hard-speaker \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_input_length_in_seconds 15 \
    --gradient_accumulation_steps 2

CUDA_VISIBLE_DEVICES=1 python asr_finetuning.py \
    --model_name_or_path EdoAbati/whisper-large-v2-it \
    --dataset_name_or_path RiTA-nlp/italic-hard-noisy \
    --batch_size 4 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --max_input_length_in_seconds 15 \
    --gradient_accumulation_steps 2


python asr_inference.py \
    --model_id models/easy/EdoAbati-whisper-large-v2-it/best_model/ \
    --dataset RiTA-nlp/italic-easy \
    --split test \
    --device 0 \
    --batch_size 8 \
    --language it >> asr_easy.txt


python asr_inference.py \
    --model_id models/speaker/EdoAbati-whisper-large-v2-it/best_model/ \
    --dataset RiTA-nlp/italic-hard-speaker \
    --split test \
    --device 0 \
    --batch_size 8 \
    --language it >> asr_hard_speaker.txt

python asr_inference.py \
    --model_id models/noisy/EdoAbati-whisper-large-v2-it/best_model/ \
    --dataset RiTA-nlp/italic-hard-noisy \
    --split test \
    --device 0 \
    --batch_size 8 \
    --language it >> asr_hard_noisy.txt