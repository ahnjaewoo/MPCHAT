## Next Response Prediction

**CLIP-CLIP**
```bash
python main_nrp.py \
  --model_type "clip-clip" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-clip/nrp/full_inputs" \
  --seed 202 \
  --sum_persona_images \
  --remove_empty_images \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_responses 100 \
  --learning_rate 3e-06 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

**CLIP-SBERT**
```bash
python main_nrp.py \
  --model_type "clip-sbert" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-sbert/nrp/full_inputs" \
  --seed 202 \
  --freeze_image_encoder \
  --sum_persona_images \
  --remove_empty_images \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_responses 100 \
  --learning_rate 1e-05 \
  --max_seq_length 128 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

## Grounding Persona Prediction

**CLIP-CLIP (no-response)**
```bash
python main_gpp.py \
  --model_type "clip-clip" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-clip/gpp-context/full_inputs" \
  --seed 202 \
  --sum_persona_images \
  --remove_empty_images \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_candidate_persona_elements 100 \
  --learning_rate 3e-06 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

**CLIP-CLIP (response)**
```bash
python main_gpp.py \
  --model_type "clip-clip" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-clip/gpp-response/full_inputs" \
  --seed 202 \
  --sum_persona_images \
  --remove_empty_images \
  --use_response \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_candidate_persona_elements 100 \
  --learning_rate 3e-06 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

**CLIP-SBERT (no-response)**
```bash
python main_gpp.py \
  --model_type "clip-sbert" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-sbert/gpp-context/full_inputs" \
  --seed 202 \
  --freeze_image_encoder \
  --sum_persona_images \
  --remove_empty_images \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_candidate_persona_elements 100 \
  --learning_rate 1e-05 \
  --max_seq_length 128 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

**CLIP-SBERT (response)**
```bash
python main_gpp.py \
  --model_type "clip-sbert" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-sbert/gpp-response/full_inputs" \
  --seed 202 \
  --freeze_image_encoder \
  --sum_persona_images \
  --remove_empty_images \
  --use_response \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_candidate_persona_elements 100 \
  --learning_rate 1e-05 \
  --max_seq_length 128 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

## Speaker Identification

**CLIP-CLIP**
```bash
python main_si.py \
  --model_type "clip-clip" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-clip/si/full_inputs" \
  --seed 202 \
  --sum_persona_images \
  --remove_empty_images \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_candidate_authors 100 \
  --learning_rate 3e-06 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```

**CLIP-SBERT**
```bash
python main_si.py \
  --model_type "clip-sbert" \
  --dialog_data_dir "." \
  --dialog_image_data_dir "./images/dialog/" \
  --persona_image_data_dir "./images/persona/" \
  --output_dir "outputs/clip-sbert/si/full_inputs" \
  --seed 202 \
  --freeze_image_encoder \
  --sum_persona_images \
  --remove_empty_images \
  --do_train \
  --do_test \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 4 \
  --max_num_candidate_authors 100 \
  --learning_rate 2e-05 \
  --max_seq_length 128 \
  --weight_decay 0.05 \
  --num_train_epochs 5 \
  --save_epoch 1 \
  --num_workers 12
```
