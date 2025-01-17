# Our standard stereo model
<<<<<<< HEAD
=======
#标准双目模型
>>>>>>> aspp/master
python ../train.py --model_name S_640x192 \
  --use_stereo --frame_ids 0 --split eigen_full

# Our low resolution stereo model
<<<<<<< HEAD
=======
#低分辨率双目模型
>>>>>>> aspp/master
python ../train.py --model_name S_416x128 \
  --use_stereo --frame_ids 0 --split eigen_full \
  --height 128 --width 416

# Our high resolution stereo model
<<<<<<< HEAD
=======
#高分辨率双目模型
>>>>>>> aspp/master
python ../train.py --model_name S_1024x320 \
  --use_stereo --frame_ids 0 --split eigen_full \
  --height 320 --width 1024 \
  --load_weights_folder ~/tmp/S_640x192/models/weights_9 \
  --models_to_load encoder depth \
  --num_epochs 5 --learning_rate 1e-5

# Our standard stereo model w/o pretraining
<<<<<<< HEAD
=======
#标准双目模型，不带有预训练
>>>>>>> aspp/master
python ../train.py --model_name S_640x192_no_pt \
  --use_stereo --frame_ids 0 --split eigen_full \
  --weights_init scratch \
  --num_epochs 30

# Baseline stereo model, i.e. ours with our contributions turned off
<<<<<<< HEAD
=======
#双目模型baseline，不带有automasking
>>>>>>> aspp/master
python ../train.py --model_name S_640x192_baseline \
  --use_stereo --frame_ids 0 --split eigen_full \
  --v1_multiscale --disable_automasking
